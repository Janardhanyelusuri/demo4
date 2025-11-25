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

interface DynamicBarChartProps {
  title: string;
  cloudPlatform: string;
  queryType: string;
  id: string;
  nameField: string;
  valueFields: string[];
  topItemsToShow?: number;
}

interface ChartItem {
  name: string;
  values: {
    [key: string]: number;
  };
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

const DynamicBarChart: React.FC<DynamicBarChartProps> = ({
  title,
  cloudPlatform,
  queryType,
  id,
  nameField,
  valueFields,
  topItemsToShow = 6,
}) => {
  const [chartData, setChartData] = useState<ChartItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const getData = async () => {
      let retries = 0;
      const maxRetries = 3;

      while (retries < maxRetries) {
        try {
          const result = await apiQueue.enqueue(
            () => fetchDashboardData(cloudPlatform, queryType, id),"DynamicBar");
          const formattedData = result.data.map((item: any) => {
            const values = valueFields.reduce((acc: any, field: string) => {
              acc[field] = parseFloat(item[field]) || 0;
              return acc;
            }, {});

            return {
              name: item[nameField],
              values,
            };
          });
          console.log("Formatted Data:", formattedData); // Log formatted data
          processChartData(formattedData);
          setLoading(false);
          return; // Exit the function if successful
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
  }, [cloudPlatform, queryType, id, nameField, valueFields]);

  const processChartData = useCallback(
    (data: ChartItem[]) => {
      const sortedData = data
        .filter((item) => Object.values(item.values).some((value) => value > 0))
        .sort(
          (a, b) =>
            Object.values(b.values).reduce((sum, v) => sum + v, 0) -
            Object.values(a.values).reduce((sum, v) => sum + v, 0)
        );

      const topItems = sortedData.slice(0, topItemsToShow - 1);
      const otherItems = sortedData.slice(topItemsToShow - 1);

      const processedData = [
        ...topItems,
        {
          name: "Others",
          values: valueFields.reduce((acc: any, field: string) => {
            acc[field] = otherItems.reduce(
              (sum, item) => sum + item.values[field],
              0
            );
            return acc;
          }, {}),
        },
      ];

      setChartData(processedData);
    },
    [topItemsToShow, valueFields]
  );

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-white p-2 border rounded shadow">
          <p className="font-semibold">{item.name}</p>
          {valueFields.map((field) => (
            <p key={field}>{`${field}: $${item.values[field].toFixed(1)}`}</p>
          ))}
        </div>
      );
    }
    return null;
  };

  if (loading)
    return (
      <div className="bg-[#fff] p-4 rounded-md shadow-md border border-[#E0E5EF] flex items-center justify-center min-h-[80px]">
        <Loader />
      </div>
    );
  if (error)
    return (
      <div className="bg-[#fff] p-4 rounded-md shadow-md border border-[#E0E5EF] flex items-center justify-center min-h-[80px] text-[#D82026]">
        <CircleAlert className="w-5 h-5 mr-2" />
        <span>Error loading data</span>
      </div>
    );

  if (chartData.length === 0)
    return (
      <div className="bg-[#fff] p-4 rounded-md shadow-md border border-[#E0E5EF] flex items-center justify-center min-h-[80px] text-[#6B7280]">
        No data available
      </div>
    );

  return (
    <div className="h-full w-full bg-[#fff] rounded-md shadow-md border border-[#E0E5EF] p-4 flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-[#233E7D]">{title}</h2>
        <BarChartIcon className="w-6 h-6 text-[#233E7D]" />
      </div>
      <div className="flex-grow">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E5EF" />
            <XAxis
              type="number"
              tickFormatter={(value) => `$${value.toFixed(0)}`}
              domain={[0, "dataMax"]}
              tick={{ fontSize: 12, fill: '#6B7280' }}
              axisLine={{ stroke: '#E0E5EF' }}
              tickLine={{ stroke: '#E0E5EF' }}
            />
            <YAxis
              dataKey="name"
              type="category"
              width={150}
              tick={{ fontSize: 12, fill: '#6B7280' }}
              axisLine={{ stroke: '#E0E5EF' }}
              tickLine={{ stroke: '#E0E5EF' }}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: '#233E7D10' }} />
            <Legend wrapperStyle={{ color: '#233E7D', fontSize: '12px' }} />
            {valueFields.map((field, index) => (
              <Bar
                key={field}
                dataKey={`values.${field}`}
                fill={COLORS[index % COLORS.length]}
                radius={[0, 4, 4, 0]}
              >
                {chartData.map((entry, entryIndex) => (
                  <Cell
                    key={`cell-${entryIndex}-${field}`}
                    fill={COLORS[entryIndex % COLORS.length]}
                  />
                ))}
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default DynamicBarChart;
