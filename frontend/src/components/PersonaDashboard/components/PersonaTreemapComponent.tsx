import React, { useEffect, useState } from "react";
import { fetchDashboardData } from "@/lib/api";
import Loader from "@/components/Loader";
import { CircleAlert, LayoutGrid } from "lucide-react";
import { Treemap, ResponsiveContainer, Tooltip, Cell } from "recharts";
import apiQueue from "@/lib/apiQueueManager";

interface PersonaTreemapComponentProps {
  title: string;
  cloudPlatform: string;
  queryType: string;
  id: string;
  nameField: string;
  valueField: string;
}

interface TreemapItem {
  name: string;
  size: number;
}

const COLORS = [
  "#3b82f6",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#ec4899",
  "#14b8a6",
  "#f97316",
  "#6366f1",
  "#84cc16",
  "#06b6d4",
  "#d946ef",
];

const PersonaTreemapComponent: React.FC<PersonaTreemapComponentProps> = ({
  title,
  cloudPlatform,
  queryType,
  id,
  nameField,
  valueField,
}) => {
  const [chartData, setChartData] = useState<TreemapItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const getData = async () => {
      let retries = 0;
      const maxRetries = 3;

      while (retries < maxRetries) {
        try {
          const result = await apiQueue.enqueue(
            () => fetchDashboardData(cloudPlatform, queryType, id),"PersonaTreeMap");
          const formattedData = result.map((item: any) => ({
            name: item[nameField],
            size: parseFloat(item[valueField]),
          }));
          setChartData(formattedData);
          setLoading(false);
          return;
        } catch (error) {
          retries++;
          if (retries === maxRetries) {
            setError(
              error instanceof Error
                ? error
                : new Error("An unknown error occurred")
            );
            setLoading(false);
          }
        }
      }
    };

    getData();
  }, [cloudPlatform, queryType, id, nameField, valueField]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-2 border rounded shadow">
          <p className="font-semibold">{payload[0].payload.name}</p>
          <p>{`Value: $${payload[0].value.toFixed(1)}`}</p>
        </div>
      );
    }
    return null;
  };

  if (loading) return <Loader />;
  if (error) return <CircleAlert color="red" />;

  return (
    <div className="h-full w-full bg-[#FFFFFF] rounded-md shadow hover:shadow-lg border border-[#E0E5EF] flex flex-col p-4 transition-all duration-300">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-[#233E7D] tracking-tight">{title}</h2>
        <LayoutGrid className="w-6 h-6 text-[#D82026]" />
      </div>
      <div className="flex-grow">
        <ResponsiveContainer width="100%" height={300}>
          <Treemap
            data={chartData}
            dataKey="size"
            aspectRatio={4 / 3}
            stroke="#fff"
            fill="#D82026"
          >
            {chartData.map((entry, index) => (
              <Tooltip key={`cell-${index}`} content={<CustomTooltip />} />
            ))}
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={COLORS[index % COLORS.length]}
              />
            ))}
          </Treemap>
        </ResponsiveContainer>
      </div>
      <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
    </div>
  );
};

export default PersonaTreemapComponent;
