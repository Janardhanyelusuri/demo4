import React, { useCallback, useEffect, useState } from "react";
import { fetchDashboardData } from "@/lib/api";
import Loader from "@/components/Loader";
import { CircleAlert, BarChart as BarChartIcon } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import apiQueue from "@/lib/apiQueueManager";

interface PersonaBarChartComponentProps {
  title: string;
  queryType: string;
  id: string;
  nameField: string;
  valueField: string;
  topItemsToShow?: number;
  tag_id?: number;
  duration?: string;
  setParentLoading?: (isLoading: boolean) => void;
}

interface ChartItem {
  name: string;
  fullName: string;
  value: number;
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

const PersonaBarChartComponent: React.FC<PersonaBarChartComponentProps> = ({
  title,
  queryType,
  id,
  nameField,
  valueField,
  topItemsToShow = 6,
  tag_id,
  duration,
  setParentLoading
}) => {
  const [chartData, setChartData] = useState<ChartItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const [noData, setNoData] = useState<boolean>(false); // New state for no data message

  useEffect(() => {
    const getData = async () => {
      setLoading(true);
      if (setParentLoading) setParentLoading(true);
      
      const maxRetries = 3;
      let retries = 0;
      let success = false;
      while (retries < maxRetries && !success) {
        try {
          const result = await apiQueue.enqueue(
            () => fetchDashboardData(queryType, id, undefined, tag_id, duration),"PersonaBarChart");
          const formattedData = result.data.map((item: any) => ({
            name: item[nameField],
            value: parseFloat(item[valueField]),
          }));
          processChartData(formattedData);
          setLoading(false);
          if (setParentLoading) setParentLoading(false);
          success = true;
        } catch (error) {
          retries++;
          if (retries === maxRetries) {
            setError(
              error instanceof Error
                ? error
                : new Error("An unknown error occurred")
            );
            setLoading(false);
            if (setParentLoading) setParentLoading(false);
          }
        }
      }
    };

    getData();
  }, [queryType, id, nameField, valueField, tag_id, duration]);

  const processChartData = useCallback(
    (data: ChartItem[]) => {
      const sortedData = data
        .filter((item) => item.value > 0)
        .sort((a, b) => b.value - a.value);

      const topItems = sortedData.slice(0, topItemsToShow - 1);
      const otherItems = sortedData.slice(topItemsToShow - 1);

      const processedData = [
        ...topItems,
        {
          name: "Others",
          value: otherItems.reduce((sum, item) => sum + item.value, 0),
        },
      ];

      const totalValue = processedData.reduce(
        (sum, item) => sum + item.value,
        0
      );

      if (totalValue === 0 || processedData.length === 0) {
        setNoData(true);
        setChartData([]);
        return;
      }

      const formattedChartData = processedData.map((item) => ({
        ...item,
        name: truncateText(item.name, 20),
        fullName: `${item.name} (${((item.value / totalValue) * 100).toFixed(
          1
        )}%)`,
      }));

      setChartData(formattedChartData);
      setNoData(false);
    },
    [topItemsToShow]
  );

  const truncateText = (text: string, maxLength: number) => {
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength - 3) + "...";
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-2 border rounded shadow">
          <p className="font-semibold">{payload[0].payload.fullName}</p>
          <p>{`Value: $${payload[0].value.toFixed(1)}`}</p>
        </div>
      );
    }
    return null;
  };

  if (loading)
    return (
      <div className="flex justify-center items-center h-full">
        <Loader />
      </div>
    );
  if (error)
    return (
      <div className="flex justify-center items-center h-full text-red-500">
        <CircleAlert className="mr-2" />
        <span>Error loading data</span>
      </div>
    );
    if (noData)
      return (
        <div className="h-full w-full bg-[#FFFFFF] rounded-md shadow hover:shadow-lg border border-[#E0E5EF] p-4 flex flex-col transition-all duration-300">
          <h2 className="text-lg font-bold text-[#233E7D] tracking-tight">{title}</h2>
          <div className="flex justify-center items-center h-full p-4 text-[#6B7280]">
            <span>Data Insufficient for Chart Visualization</span>
          </div>
        </div>
      );

  return (
    <div className="h-full w-full bg-[#FFFFFF] rounded-md shadow hover:shadow-lg border border-[#E0E5EF] flex flex-col p-4 transition-all duration-300">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-[#233E7D] tracking-tight">{title}</h2>
        <BarChartIcon className="w-6 h-6 text-[#D82026]" />
      </div>
      <div className="flex-grow flex justify-center items-center">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 5, right: 30, left: -40, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E5EF" />
            <XAxis
              type="number"
              tickFormatter={(value) => `$${value.toFixed(0)}`}
              domain={[0, "dataMax"]}
              axisLine={{ stroke: '#C8C8C8' }}
              tick={{ fill: '#233E7D', fontWeight: 500, fontSize: 12 }}
            />
            <YAxis
              dataKey="name"
              type="category"
              width={150}
              axisLine={{ stroke: '#C8C8C8' }}
              tick={{ fill: '#233E7D', fontWeight: 500, fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} wrapperStyle={{ background: '#fff', border: '1px solid #E0E5EF', borderRadius: '0.375rem', color: '#233E7D' }} labelStyle={{ color: '#233E7D', fontWeight: 600 }} itemStyle={{ color: '#233E7D' }} />
            <Legend wrapperStyle={{ color: '#233E7D', fontWeight: 500, fontSize: 12 }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
    </div>
  );
};

export default PersonaBarChartComponent;
