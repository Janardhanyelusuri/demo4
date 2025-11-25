import React, { useEffect, useState } from "react";
import { fetchDashboardData } from "@/lib/api";
import Loader from "@/components/Loader";
import { CircleAlert, LineChart as LineChartIcon } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

import apiQueue from "@/lib/apiQueueManager";

interface PersonaLineChartComponentProps {
  title: string;
  queryType: string;
  id: string;
  xKey: string;
  yKey: string;
  name: string;
  tag_id?: number; // Optional tag_id
  duration?: string; // Optional duration parameter
  setParentLoading?: (isLoading: boolean) => void;
  xAxisLabelFormatter?: (label: string) => string;
  xAxisTooltipFormatter?: (label: string) => string;
}

const granularityOptions = [
  { value: "day", label: "Daily" },
  { value: "week", label: "Weekly" },
  { value: "month", label: "Monthly" },
  { value: "quarter", label: "Quarterly" },
];

const PersonaLineChartComponent: React.FC<PersonaLineChartComponentProps> = ({
  title,
  queryType,
  id,
  xKey,
  yKey,
  name,
  tag_id, // Ensure tag_id is passed to the component
  duration, // Add duration parameter
  setParentLoading, // Add setParentLoading callback
  xAxisLabelFormatter = (label: string) => label,
  xAxisTooltipFormatter = (label: string) => label,
}) => {
  const [chartData, setChartData] = useState<any[]>([]);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [noData, setNoData] = useState<boolean>(false);
  const [granularity, setGranularity] = useState<string>("week"); // Default to weekly

  useEffect(() => {
    const getData = async () => {
      setLoading(true);
      if (setParentLoading) setParentLoading(true);
      
      console.log("PersonaLineChart - API call with:", { queryType, id, granularity, tag_id, duration });
      
      let retries = 0;
      const maxRetries = 3;
      while (retries < maxRetries) {
        try {
          // Pass tag_id and duration to fetchDashboardData if available
          const result = await apiQueue.enqueue(
            () => fetchDashboardData(queryType, id, granularity, tag_id, duration), // Pass duration here
            "PersonaLineChart"
          );
          const filteredData = result.data.filter(
            (item: any) => item[yKey] > 0
          );

          setChartData(filteredData);
          setLoading(false);
          if (setParentLoading) setParentLoading(false);
          setNoData(filteredData.length === 0);
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
            if (setParentLoading) setParentLoading(false);
          }
        }
      }
    };

    getData();
  }, [queryType, id, granularity, tag_id, duration]); // Add duration to dependencies

  const handleGranularityChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    setGranularity(event.target.value);
    setLoading(true); // Show loader on granularity change
    if (setParentLoading) setParentLoading(true);
  };

  if (loading) return <Loader />;
  if (error) return <CircleAlert color="red" />;
  if (noData)
    return (
      <div className="h-full w-full bg-[#FFFFFF] rounded-md shadow hover:shadow-lg border border-[#E0E5EF] p-4 flex flex-col transition-all duration-300">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-[#233E7D] tracking-tight">{title}</h2>
          <div className="flex items-center justify-right gap-3">
            <div className="mb-4">
              <select
                aria-label="Select granularity"
                value={granularity}
                onChange={handleGranularityChange}
                className="p-2 border border-[#C8C8C8] rounded-md text-[#233E7D] focus:ring-2 focus:ring-[#233E7D]"
              >
                {granularityOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <LineChartIcon className="w-6 h-6 text-[#D82026]" />
          </div>
        </div>
        <div className="flex justify-center items-center h-full p-4 text-[#6B7280]">
          <span>Data Insufficient for Chart Visualization</span>
        </div>
      </div>
    );

  const minY = Math.min(...chartData.map((item) => item[yKey]));
  const maxY = Math.max(...chartData.map((item) => item[yKey]));
  const yAxisPadding = (maxY - minY) * 0.1; // Add 10% padding

  return (
    <div className="h-full w-full bg-[#FFFFFF] rounded-md shadow hover:shadow-lg border border-[#E0E5EF] flex flex-col p-4 transition-all duration-300">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-[#233E7D] tracking-tight">{title}</h2>
        <div className="flex items-center justify-right gap-3">
          <div className="mb-4">
            <select
              aria-label="Select granularity"
              value={granularity}
              onChange={handleGranularityChange}
              className="p-2 border border-[#C8C8C8] rounded-md text-[#233E7D] focus:ring-2 focus:ring-[#233E7D]"
            >
              {granularityOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <LineChartIcon className="w-6 h-6 text-[#D82026]" />
        </div>
      </div>
      <div className="flex-grow flex items-center justify-center">
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart
            data={chartData}
            margin={{ top: -30, right: 30, left: 50, bottom: -10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E5EF" />
            <XAxis
              dataKey={xKey}
              tickFormatter={xAxisLabelFormatter}
              axisLine={{ stroke: '#C8C8C8' }}
              tick={{ fill: '#233E7D', fontWeight: 500, fontSize: 12 }}
            />
            <YAxis
              axisLine={{ stroke: '#C8C8C8' }}
              tick={{ fill: '#233E7D', fontWeight: 500, fontSize: 12 }}
              domain={[minY - yAxisPadding, maxY + yAxisPadding]}
              tickFormatter={(value) => `$${value.toLocaleString()}`}
              width={60}
            />
            <Tooltip
              labelFormatter={(label) => {
                const date = new Date(label);
                return date.toLocaleDateString();
              }}
              formatter={(value) => {
                if (typeof value === "number") {
                  return [`$${value.toFixed(2)}`];
                }
                return [`${value}`];
              }}
              contentStyle={{
                backgroundColor: "#fff",
                border: "1px solid #E0E5EF",
                borderRadius: "0.375rem",
                color: "#233E7D"
              }}
              labelStyle={{ color: '#233E7D', fontWeight: 600 }}
              itemStyle={{ color: '#233E7D' }}
            />
            <defs>
              <linearGradient id="colorUv" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#233E7D" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#233E7D" stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area
              type="monotone"
              dataKey={yKey}
              stroke="#D82026"
              fillOpacity={1}
              fill="url(#colorUv)"
              name={name}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
    </div>
  );
};

export default PersonaLineChartComponent;
