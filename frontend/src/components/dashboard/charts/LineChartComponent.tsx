import React, { useEffect, useState, useRef } from "react";
import { fetchData, fetchResourceNames, fetchResourceIds } from "@/lib/api";
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

interface ResourceNameResult {
  id?: string;
  name?: string;
  resourceId?: string;
  displayName?: string;
}

interface LineChartComponentProps {
  title: string;
  cloudPlatform: string;
  queryType: string;
  projectId: string;
  service_names?: string;
  xKey: string;
  yKey: string;
  name: string;
  xAxisLabelFormatter?: (label: string) => string;
  xAxisTooltipFormatter?: (label: string) => string;
  granularity?: string;
  tag_id?: number; // Added tag_id as an optional prop
  duration?: string; // Optional duration parameter
  setParentLoading?: (isLoading: boolean) => void;
}

const granularityOptions = [
  { value: "day", label: "Daily" },
  { value: "week", label: "Weekly" },
  { value: "month", label: "Monthly" },
  { value: "quarter", label: "Quarterly" },
];

const LineChartComponent: React.FC<LineChartComponentProps> = ({
  title,
  cloudPlatform,
  queryType,
  projectId,
  service_names,
  xKey,
  yKey,
  name,
  xAxisLabelFormatter = (label: string) => label,
  xAxisTooltipFormatter = (label: string) => label,
  tag_id, // Added tag_id to the destructured props
  duration,
  setParentLoading,
}) => {
  const [chartData, setChartData] = useState<any[]>([]);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [noData, setNoData] = useState<boolean>(false);
  const [granularity, setGranularity] = useState<string>("day");
  const hasFetched = useRef(false);

  useEffect(() => {
    if (hasFetched.current) return;
    hasFetched.current = true;
    if (setParentLoading) setParentLoading(true);
    const getData = async () => {
      let retries = 0;
      const maxRetries = 3;
      let resourceNameParam: string | undefined;
      while (retries < maxRetries) {
        try {
          let resourceNameParam: string | undefined;
          if (tag_id) {
            let resourceResults;
            if (cloudPlatform === "aws") {
              resourceResults = await apiQueue.enqueue(() => fetchResourceIds(tag_id), "ResourceIds");
            } else {
              resourceResults = await apiQueue.enqueue(() => fetchResourceNames(tag_id), "ResourceNames");
            }
            const firstResource = resourceResults?.[0];
            if (firstResource && typeof firstResource === "object") {
              resourceNameParam = firstResource.name ?? firstResource.id;
            }
          }
          const result = await apiQueue.enqueue(
            () => fetchData(cloudPlatform, queryType, projectId, granularity, resourceNameParam, service_names, tag_id, duration),
            "FetchData"
          );
          const filteredData = result.data.filter((item: any) => item[yKey] > 0);
          setChartData(filteredData);
          setLoading(false);
          setNoData(filteredData.length < 0);
          if (setParentLoading) setParentLoading(false);
          return;
        } catch (error) {
          retries++;
          if (retries === maxRetries) {
            setError(error instanceof Error ? error : new Error("An unknown error occurred"));
            setLoading(false);
            if (setParentLoading) setParentLoading(false);
          }
        }
      }
    };
    getData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cloudPlatform, queryType, projectId, xKey, yKey, granularity, tag_id, duration, service_names]);

  
  const handleGranularityChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setGranularity(event.target.value);
    setLoading(true); // Show loader on granularity change
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
        <span>Error loading chart</span>
      </div>
    );
  if (noData)
    return (
      <div className="h-full w-full bg-[#fff] rounded-md shadow-md border border-[#E0E5EF] p-4 flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-[#233E7D]">{title}</h2>
          <div className="flex items-center justify-right gap-3">
            <label htmlFor="granularity" className="mr-2 text-[#6B7280]">
              {/* Granularity: */}
            </label>
            <select
              id="granularity"
              value={granularity}
              onChange={handleGranularityChange}
              className="border border-[#E0E5EF] rounded-md p-1 text-[#233E7D] focus:ring-2 focus:ring-[#233E7D]"
            >
              {granularityOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <LineChartIcon className="w-6 h-6 text-[#233E7D]" />
          </div>
        </div>
        <div className="flex justify-center items-center h-full p-4 text-[#6B7280]">
          <span>Data Insufficient for Chart Visualization</span>
        </div>
      </div>
    );

  const latestValue = chartData[chartData.length - 1]?.[yKey] || 0;
  const previousValue = chartData[chartData.length - 2]?.[yKey] || 0;
  const percentageChange =
    ((latestValue - previousValue) / previousValue) * 100;

  const formatCurrency = (value: number) =>
    new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(value);
    // Added this compact date formatter
    const compactDateFormatter = (label: string) => {
      try {
        const date = new Date(label);
        
        if (granularity === 'day') {
          // Changed to date/month/year format
          return `${date.getDate()}/${date.getMonth() + 1}/${date.getFullYear().toString().slice(-2)}`;
        } else if (granularity === 'week') {
          // Changed to date/month/year format
          return `${date.getDate()}/${date.getMonth() + 1}/${date.getFullYear().toString().slice(-2)}`;
        } else if (granularity === 'month') {
          // Monthly: Show "MMM 'YY"
          return date.toLocaleDateString('en-US', { month: 'short' }) + ` '${date.getFullYear().toString().slice(-2)}`;
        } else {
          // Quarterly: Show "Qx 'YY"
          const quarter = Math.floor(date.getMonth() / 3) + 1;
          return `Q${quarter} '${date.getFullYear().toString().slice(-2)}`;
        }
      } catch {
        return label;
      }
    };

  return (
    <div className="h-full w-full bg-[#fff] rounded-md shadow-md border border-[#E0E5EF] p-4 flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-[#233E7D]">{title}</h2>
        <div className="flex items-center justify-right gap-3">
          <label htmlFor="granularity" className="mr-2 text-[#6B7280]">
            {/* Granularity: */}
          </label>
          <select
            id="granularity"
            value={granularity}
            onChange={handleGranularityChange}
            className="border border-[#E0E5EF] rounded-md p-1 text-[#233E7D] focus:ring-2 focus:ring-[#233E7D]"
          >
            {granularityOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <LineChartIcon className="w-6 h-6 text-[#233E7D]" />
        </div>
      </div>

      <div className="flex-grow flex items-center justify-center">
        <ResponsiveContainer width="100%" height={350}>
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 10, bottom: 15 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E5EF" />
            <XAxis
              dataKey={xKey}
              tickFormatter={compactDateFormatter}
              stroke="#6B7280"
              height={50}
              tick={{ fontSize: 10, fill: '#6B7280' }}
              interval="preserveStartEnd"
              axisLine={{ stroke: '#E0E5EF' }}
              tickLine={{ stroke: '#E0E5EF' }}
            />
            <YAxis
              stroke="#6B7280"
              tickFormatter={(value) => `$${value}`}
              tick={{ fontSize: 10, fill: '#6B7280' }}
              axisLine={{ stroke: '#E0E5EF' }}
              tickLine={{ stroke: '#E0E5EF' }}
            />
            <Tooltip
              offset={20}
              labelFormatter={(label) => {
                const date = new Date(label);
                return date.toLocaleDateString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric',
                });
              }}
              formatter={(value) => {
                if (typeof value === "number") {
                  return [`$${value.toFixed(2)}`];
                }
                return [`${value}`];
              }}
              contentStyle={{
                backgroundColor: "#fff",
                border: '1px solid #E0E5EF',
                borderRadius: "6px",
                padding: "8px",
                boxShadow: "0 2px 10px rgba(35,62,125,0.08)",
                color: '#233E7D',
                fontWeight: 500,
              }}
              itemStyle={{ color: '#233E7D' }}
              labelStyle={{ color: '#6B7280' }}
              cursor={{ fill: '#233E7D10' }}
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
              stroke="#233E7D"
              fillOpacity={1}
              fill="url(#colorUv)"
              name={name}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default LineChartComponent;
