import React, { useCallback, useEffect, useState, useRef } from "react";
import { fetchData, fetchResourceNames, fetchResourceIds } from "@/lib/api";
import Loader from "@/components/Loader";
import { CircleAlert, PieChart as PieChartIcon } from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
  Sector,
  LabelList,
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

interface PieChartComponentProps {
  title: string;
  cloudPlatform: string;
  queryType: string;
  projectId: string;
  nameField: string;
  valueField: string;
  tag_id?: number;
  duration?: string; // Optional duration parameter
  topItemsToShow?: number;
  service_names?: string;
  granularity?: string;
  setParentLoading?: (isLoading: boolean) => void;
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

// remove any black label from inside chart and handle custom active highlight only
const renderActiveShape = (props: any) => {
  const {
    cx,
    cy,
    innerRadius,
    outerRadius,
    startAngle,
    endAngle,
    fill,
  } = props;

  return (
    <Sector
      cx={cx}
      cy={cy}
      innerRadius={innerRadius}
      outerRadius={outerRadius + 10}
      startAngle={startAngle}
      endAngle={endAngle}
      fill={fill}
    />
  );
};

const PieChartComponent: React.FC<PieChartComponentProps> = ({
  title,
  cloudPlatform,
  queryType,
  projectId,
  nameField,
  valueField,
  tag_id,
  duration,
  topItemsToShow = 6,
  service_names,
  granularity: initialGranularity = "",
  setParentLoading,
}) => {
  const [chartData, setChartData] = useState<ChartItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const [noData, setNoData] = useState<boolean>(false);
  const [granularity, setGranularity] = useState<string>(initialGranularity);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const hasFetched = useRef(false);

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
        value: Number(item.value.toFixed(1)),
      }));

      setChartData(formattedChartData);
      setNoData(false);
    },
    [topItemsToShow]
  );

  const fetchDataAndUpdateState = useCallback(async () => {
    if (setParentLoading) setParentLoading(true);
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
  }, [
    cloudPlatform,
    queryType,
    projectId,
    nameField,
    valueField,
    tag_id,
    duration,
    service_names,
    processChartData,
    granularity,
  ]);

  useEffect(() => {
    if (hasFetched.current) return;
    hasFetched.current = true;
    fetchDataAndUpdateState();
  }, [fetchDataAndUpdateState]);

  const handleGranularityChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    setGranularity(event.target.value);
    setLoading(true);
  };

  const handleSliceClick = (index: number) => {
    setActiveIndex(index === activeIndex ? null : index);
  };

  const handleLegendClick = (_: any, index: number) => {
    handleSliceClick(index);
  };

  const truncateText = (text: string, maxLength: number) => {
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength - 3) + "...";
  };

  const CustomTooltip = ({ active, payload }: any) => {
    const index = activeIndex;
    if (index !== null && chartData[index]) {
      const item = chartData[index];
      return (
        <div className="bg-white p-2 border rounded shadow text-xs">
          <p className="font-semibold">{item.fullName}</p>
          <p>{`Value: $${item.value.toFixed(1)}`}</p>
        </div>
      );
    }

    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-2 border rounded shadow text-xs">
          <p className="font-semibold">{payload[0].payload.fullName}</p>
          <p>{`Value: $${payload[0].value.toFixed(1)}`}</p>
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
  if (noData)
    return (
      <div className="h-full w-full bg-[#fff] rounded-md shadow-md border border-[#E0E5EF] p-4 flex flex-col">
        <h2 className="text-lg font-semibold text-[#233E7D]">{title}</h2>
        <div className="flex justify-center items-center h-full p-4 text-[#6B7280]">
          <span>Data Insufficient for Chart Visualization</span>
        </div>
      </div>
    );

  return (
    <div className="h-full w-full bg-[#fff] rounded-md shadow-md border border-[#E0E5EF] p-2 flex flex-col">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg font-semibold text-[#233E7D]">{title}</h2>
        <div className="flex items-center gap-3">
          <PieChartIcon className="w-4 h-4 text-[#233E7D]" />
        </div>
      </div>
      <div className="flex-grow relative">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              activeIndex={activeIndex ?? undefined}
              activeShape={renderActiveShape}
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius="80%"
              fill="#233E7D"
              dataKey="value"
              onClick={(_, index) => handleSliceClick(index)}
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                />
              ))}
              <LabelList
                dataKey="value"
                position="inside"
                formatter={(val: number, entry: any) => {
                  const total = chartData.reduce((sum, item) => sum + item.value, 0);
                  const percent = (val / total) * 100;
                  return `${percent.toFixed(0)}%`;
                }}
                style={{
                  fill: "white",
                  fontSize: "10px",
                }}
              />
            </Pie>
            <Tooltip content={<CustomTooltip />} cursor={{ fill: '#233E7D10' }} />
            <Legend
              layout="vertical"
              align="right"
              verticalAlign="middle"
              onClick={handleLegendClick}
              formatter={(value: string, entry: any, index: number) => {
                const item = chartData[index];
                return (
                  <span
                    style={{
                      color: activeIndex === index ? "#233E7D" : "#6B7280",
                      fontWeight: "normal",
                      cursor: "pointer",
                      fontSize: "10px",
                    }}
                    title={item?.fullName || value}
                  >
                    {value}
                  </span>
                );
              }}
              wrapperStyle={{
                fontSize: "10px",
                paddingLeft: "10px",
                right: 0,
                maxWidth: "40%",
                overflowY: "auto",
                color: '#233E7D',
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PieChartComponent;
