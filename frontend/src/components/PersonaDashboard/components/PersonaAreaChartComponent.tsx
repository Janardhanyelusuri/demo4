import React, { useCallback, useEffect, useState } from "react";
import { fetchDashboardData } from "@/lib/api";
import Loader from "@/components/Loader";
import { CircleAlert, AreaChart as AreaChartIcon } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import apiQueue from "@/lib/apiQueueManager";

interface PersonaAreaChartComponentProps {
  title: string;
  cloudPlatform: string;
  queryType: string;
  id: string;
  nameField: string;
  valueField: string;
  topItemsToShow?: number;
}

interface ChartItem {
  name: string;
  value: number;
  fullName: string;
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

const PersonaAreaChartComponent: React.FC<PersonaAreaChartComponentProps> = ({
  title,
  cloudPlatform,
  queryType,
  id,
  nameField,
  valueField,
  topItemsToShow = 6,
}) => {
  const [chartData, setChartData] = useState<ChartItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  const processChartData = useCallback(
    (data: ChartItem[]) => {
      const sortedData = data
        .filter((item) => item.value > 0)
        .sort((a, b) => b.value - a.value);

      const topItems = sortedData.slice(0, topItemsToShow);

      const totalValue = topItems.reduce((sum, item) => sum + item.value, 0);

      const formattedChartData = topItems.map((item) => ({
        ...item,
        name: truncateText(item.name, 20),
        fullName: `${item.name} (${((item.value / totalValue) * 100).toFixed(
          1
        )}%)`,
        value: Number(item.value.toFixed(1)),
      }));

      setChartData(formattedChartData);
    },
    [topItemsToShow]
  );

  useEffect(() => {
    const getData = async () => {
      let retries = 0;
      const maxRetries = 3;

      while (retries < maxRetries) {
        try {
          const result = await apiQueue.enqueue(
            () => fetchDashboardData(cloudPlatform, queryType, id),"PersonaAreaChart");
          const formattedData = result.data.map((item: any) => ({
            name: item[nameField],
            value: parseFloat(item[valueField]),
          }));
          processChartData(formattedData);
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
  }, [cloudPlatform, queryType, id, nameField, valueField, processChartData]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-2 border rounded shadow">
          <p className="font-semibold text-xs">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-xs" style={{ color: entry.color }}>
              {`${entry.name}: $${entry.value.toFixed(1)}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const truncateText = (text: string, maxLength: number) => {
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength - 3) + "...";
  };

  if (loading)
    return (
      <div className="h-full w-full flex items-center justify-center">
        <Loader />
      </div>
    );
  if (error)
    return (
      <div className="h-full w-full flex items-center justify-center text-red-500">
        <CircleAlert size={24} />
        <span className="ml-2 text-sm">Error loading data</span>
      </div>
    );

  return (
    <div className="h-full w-full bg-[#FFFFFF] rounded-md shadow hover:shadow-lg border border-[#E0E5EF] flex flex-col p-4 transition-all duration-300">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-[#233E7D] tracking-tight">{title}</h2>
        <AreaChartIcon className="w-6 h-6 text-[#D82026]" />
      </div>
      <div className="flex-grow relative">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E5EF" />
            <XAxis dataKey="name" tick={{ fill: '#233E7D', fontWeight: 500, fontSize: 12 }} axisLine={{ stroke: '#C8C8C8' }} />
            <YAxis tick={{ fill: '#233E7D', fontWeight: 500, fontSize: 12 }} axisLine={{ stroke: '#C8C8C8' }} />
            <Tooltip content={<CustomTooltip />} wrapperStyle={{ background: '#fff', border: '1px solid #E0E5EF', borderRadius: '0.375rem', color: '#233E7D' }} labelStyle={{ color: '#233E7D', fontWeight: 600 }} itemStyle={{ color: '#233E7D' }} />
            <Legend wrapperStyle={{ color: '#233E7D', fontWeight: 500, fontSize: 12 }} />
            {chartData.map((entry, index) => (
              <Area
                key={entry.name}
                type="monotone"
                dataKey="value"
                name={entry.name}
                stackId="1"
                stroke={COLORS[index % COLORS.length]}
                fill={COLORS[index % COLORS.length]}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
    </div>
  );
};

export default PersonaAreaChartComponent;
