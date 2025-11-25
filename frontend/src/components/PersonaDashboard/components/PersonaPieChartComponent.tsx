import React, { useCallback, useEffect, useState } from "react";
import { fetchDashboardData } from "@/lib/api";
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
} from "recharts";

interface PersonaPieChartComponentProps {
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

import apiQueue from "@/lib/apiQueueManager";

interface ChartItem {
  name: string;
  value: number;
  fullName: string;
}

const COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899",
  "#14b8a6", "#f97316", "#6366f1", "#84cc16", "#06b6d4", "#d946ef",
];

const PersonaPieChartComponent: React.FC<PersonaPieChartComponentProps> = ({
  title, queryType, id, nameField, valueField, topItemsToShow = 6, tag_id, duration, setParentLoading,
}) => {
  const [chartData, setChartData] = useState<ChartItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const [noData, setNoData] = useState<boolean>(false);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

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

      const totalValue = processedData.reduce((sum, item) => sum + item.value, 0);

      if (totalValue === 0 || processedData.length === 0) {
        setNoData(true);
        setChartData([]);
        return;
      }

      const formattedChartData = processedData.map((item) => ({
        ...item,
        name: truncateText(item.name, 20),
        fullName: `${item.name} (${((item.value / totalValue) * 100).toFixed(1)}%)`,
        value: Number(item.value.toFixed(1)),
      }));

      setChartData(formattedChartData);
      setNoData(false);
    },
    [topItemsToShow]
  );

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
            () => fetchDashboardData(queryType, id, undefined, tag_id, duration),
            "PersonaPieChart"
          );
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
  }, [queryType, id, nameField, valueField, tag_id, duration, processChartData]);

  const renderCustomizedLabel = ({
    cx,
    cy,
    midAngle,
    innerRadius,
    outerRadius,
    percent,
  }: any) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text
        x={x}
        y={y}
        fill="white"
        textAnchor={x > cx ? "start" : "end"}
        dominantBaseline="central"
        fontSize="10"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  const renderActiveShape = (props: any) => {
    const {
      cx, cy, innerRadius, outerRadius,
      startAngle, endAngle, fill,
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
        stroke="none"
      />
    );
  };

  const onPieEnter = (_: any, index: number) => {
    setActiveIndex(index);
  };

  const onLegendClick = (_: any, index: number) => {
    setActiveIndex(index);
  };

  const renderColorfulLegendText = (value: string, entry: any, index: number) => (
    <span
      style={{ color: "black", fontSize: "10px", cursor: "pointer" }}
      title={entry.payload.fullName}
      onMouseEnter={() => setActiveIndex(index)}
      onClick={() => setActiveIndex(index)}
    >
      {value}
    </span>
  );

  const CustomTooltip = ({ active, payload }: any) => {
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

  const truncateText = (text: string, maxLength: number) =>
    text.length <= maxLength ? text : text.slice(0, maxLength - 3) + "...";

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
    <div className="h-full w-full bg-[#FFFFFF] rounded-md shadow hover:shadow-lg border border-[#E0E5EF] flex flex-col p-2 transition-all duration-300">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg font-bold text-[#233E7D] tracking-tight">{title}</h2>
        <PieChartIcon className="w-4 h-4 text-[#D82026]" />
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
              label={renderCustomizedLabel}
              outerRadius="80%"
              fill="#D82026"
              dataKey="value"
              onMouseEnter={onPieEnter}
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} wrapperStyle={{ background: '#fff', border: '1px solid #E0E5EF', borderRadius: '0.375rem', color: '#233E7D' }} labelStyle={{ color: '#233E7D', fontWeight: 600 }} itemStyle={{ color: '#233E7D' }} />
            <Legend
              layout="vertical"
              align="right"
              verticalAlign="middle"
              formatter={renderColorfulLegendText}
              wrapperStyle={{
                fontSize: "10px",
                paddingLeft: "5px",
                right: 0,
                maxWidth: "40%",
                overflowY: "auto",
                color: '#233E7D',
                fontWeight: 500,
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
    </div>
  );
};

export default PersonaPieChartComponent;
