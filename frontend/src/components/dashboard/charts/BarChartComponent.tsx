import React, { useCallback, useEffect, useState } from "react";
import { fetchData, fetchResourceNames, fetchResourceIds } from "@/lib/api";
import Loader from "@/components/Loader";
import { CircleAlert, BarChart as BarChartIcon } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import apiQueue from "@/lib/apiQueueManager";

interface ResourceNameResult {
  id?: string;
  name?: string;
  resourceId?: string;
  displayName?: string;
}

const granularityOptions = [
  { value: "day", label: "Daily" },
  { value: "week", label: "Weekly" },
  { value: "month", label: "Monthly" },
  { value: "quarter", label: "Quarterly" },
  { value: "year", label: "Yearly" },
];

interface BarChartComponentProps {
  title: string;
  cloudPlatform: string;
  queryType: string;
  projectId: string;
  nameField: string;
  valueField: string;
  topItemsToShow?: number;
  service_names?: string;
  tag_id?: number;
  duration?: string; // Optional duration parameter
  granularity?: string;
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

const BarChartComponent: React.FC<BarChartComponentProps> = ({
  title,
  cloudPlatform,
  queryType,
  projectId,
  nameField,
  valueField,
  topItemsToShow = 6,
  service_names,
  tag_id,
  duration, // Optional duration parameter
  granularity: initialGranularity = "",
  setParentLoading,
}) => {
  const [chartData, setChartData] = useState<ChartItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const [noData, setNoData] = useState<boolean>(false);
  const [granularity, setGranularity] = useState<string>(initialGranularity);

  const handleGranularityChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    setGranularity(event.target.value);
    setLoading(true);
  };

  useEffect(() => {
    if (setParentLoading) setParentLoading(true);
    const getData = async () => {
      let retries = 0;
      const maxRetries = 3;
      while (retries < maxRetries) {
        try {
          let resourceNameParam: string | undefined;
          if (tag_id) {
            let resourceResults;
            if (cloudPlatform === "aws") {
              resourceResults = await apiQueue.enqueue(
                () => fetchResourceIds(tag_id),
                "ResourceIds"
              );
            } else {
              resourceResults = await apiQueue.enqueue(
                () => fetchResourceNames(tag_id),
                "ResourceNames"
              );
            }
            resourceNameParam =
              resourceResults?.[0]?.name || resourceResults?.[0]?.id;
          }
          const result = await apiQueue.enqueue(
            () =>
              fetchData(
                cloudPlatform,
                queryType,
                projectId,
                granularity,
                resourceNameParam,
                service_names,
                tag_id,
                duration
              ),
            "FetchData"
          );
          const formattedData = result.data.map((item: any) => ({
            name: item[nameField],
            value: parseFloat(item[valueField]),
          }));
          processChartData(formattedData);
          setLoading(false);
          if (setParentLoading) setParentLoading(false);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    cloudPlatform,
    queryType,
    projectId,
    nameField,
    valueField,
    tag_id,
    duration,
    service_names,
    granularity,
  ]);

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
      <div className="flex justify-center items-center h-full p-4">
        <Loader />
      </div>
    );
  if (error)
    return (
      <div className="flex justify-center items-center h-full p-4 text-red-500">
        <CircleAlert className="mr-2" />
        <span>Error loading data</span>
      </div>
    );
  if (noData)
    return (
      <div className="h-full w-full bg-white rounded-lg shadow-lg border border-gray-200 p-4 flex flex-col">
        <h2 className="text-lg font-semibold text-gray-700">{title}</h2>
        <div className="flex justify-center items-center h-full p-4 text-gray-500">
          <span>Data Insufficient for Chart Visualization</span>
        </div>
      </div>
    );

  return (
    <div className="h-full w-full bg-white rounded-lg shadow-lg border border-gray-200 p-4 flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-700">{title}</h2>
        <div className="flex items-center gap-3">
          <label htmlFor="granularity-select" className="sr-only">
            Select granularity
          </label>
          <BarChartIcon className="w-6 h-6 text-blue-500" />
        </div>
      </div>
      <div className="flex-grow flex justify-center items-center">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 5, right: 30, left: -50, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis
              type="number"
              tickFormatter={(value) => `$${value.toFixed(0)}`}
              domain={[0, "dataMax"]}
            />
            <YAxis
              dataKey="name"
              type="category"
              width={150}
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="value" fill="#8884d8" radius={[0, 4, 4, 0]}>
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
    </div>
  );
};

export default BarChartComponent;
