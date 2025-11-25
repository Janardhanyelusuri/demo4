"use client";
import React, { useEffect, useState, useRef } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { fetchdDataDashboard } from "@/lib/api";
import Loader from "@/components/Loader";
import { CircleAlert, BarChart as BarChartIcon } from "lucide-react";
import { apiBaseUrl } from "next-auth/client/_utils";
import apiQueue from "@/lib/apiQueueManager";

type DataItem = {
  service_name: string;
  costs: {
    month_to_date: number | null | string;  // Updated to handle string values
    quarter_to_date: string | null;
    year_to_date: string | null;
  };
};

type TooltipData = DataItem & {
  originalQuarterToDate: number;
};

type Props = {
  queryType?: string;
  id?: string;
  title: string;
  tag_id?: number;
};

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

// Threshold value to filter out negligible values (values below this will be treated as 0)
const MINIMUM_THRESHOLD = 0.01;

const StackedBarChart: React.FC<Props> = ({ queryType, id, title, tag_id }) => {
  const [data, setData] = useState<DataItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const [noData, setNoData] = useState<boolean>(false);
  const dataFetchedRef = useRef(false);

  useEffect(() => {
    const getData = async () => {
      if (dataFetchedRef.current) return;
      dataFetchedRef.current = true;

      const maxRetries = 3;
      let retries = 0;

      while (retries < maxRetries) {
        try {
          if (queryType && id) {
            const result = await apiQueue.enqueue(
              () => fetchdDataDashboard(queryType, id.toString(), tag_id),
              "StackedBarChart"
            );
            console.log("Fetched Data:", result); // Log fetched data

            if (!result || result.length === 0) {
              setNoData(true);
            } else {
              setData(result);
            }

            setLoading(false);
            return;
          } else {
            throw new Error("Missing required parameters");
          }
        } catch (error) {
          retries++;
          if (retries === maxRetries) {
            setError(error instanceof Error ? error : new Error("An unknown error occurred"));
            setLoading(false);
          }
        }
      }
    };

    getData();
  }, [queryType, id, tag_id]);

  /**
   * Convert any value to a number and check if it's above the threshold.
   * Returns 0 if the value is null, not a number, or below the threshold.
   */
  const parseAndCheckThreshold = (value: string | number | null | undefined): number => {
    if (value === null || value === undefined) return 0;
    
    // Convert string to number if needed
    const numValue = typeof value === 'string' ? parseFloat(value) : value;
    
    // Check if it's a valid number and above threshold
    if (!isNaN(numValue) && numValue >= MINIMUM_THRESHOLD) {
      return numValue;
    }
    return 0;
  };

  /**
   * Process the data for the chart.
   * Sorts by the sum of "Month to Date", "Quarter to Date", and "Year to Date" in descending order.
   * Applies a minimum threshold to filter out negligible values.
   */
  const processChartData = (data: DataItem[]) => {
    return data
      .map((item) => {
        // Handle month to date - could be string, number, or null
        let monthToDateRaw = item.costs.month_to_date;
        let monthToDate = typeof monthToDateRaw === 'string' 
          ? parseAndCheckThreshold(monthToDateRaw)
          : parseAndCheckThreshold(monthToDateRaw);
        
        // Handle quarter_to_date
        let quarterToDate = parseAndCheckThreshold(item.costs.quarter_to_date);
        
        // Handle year_to_date
        let yearToDate = parseAndCheckThreshold(item.costs.year_to_date);
  
        // Remove the condition that sets quarterToDate to 0
        // Just show the actual values for all time periods
  
        return {
          name: item.service_name,
          "Month to Date": monthToDate,
          "Quarter to Date": quarterToDate,
          "Year to Date": yearToDate,
          showQuarterToDate: quarterToDate > 0, // Show quarter to date whenever it has a value
          areValuesEqual: quarterToDate === yearToDate && quarterToDate !== 0,
        };
      })
      // Rest of your filtering and sorting logic
      .filter(item => 
        item["Month to Date"] > 0 || 
        item["Quarter to Date"] > 0 || 
        item["Year to Date"] > 0
      )
      .sort((a, b) => {
        const sumA = a["Month to Date"] + a["Quarter to Date"] + a["Year to Date"];
        const sumB = b["Month to Date"] + b["Quarter to Date"] + b["Year to Date"];
        return sumA - sumB;
      });          
  };

  const chartData = processChartData(data);

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const itemData = chartData.find(item => item.name === label);
    
    return (
      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
        <p className="font-semibold text-gray-800">{label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-gray-600">
            {`${entry.name}: $${entry.value.toFixed(2)}`}
            {entry.name === "Quarter to Date" && itemData?.areValuesEqual && (
              <span className="text-gray-400"> (Same as Year to Date)</span>
            )}
            {entry.name === "Year to Date" && itemData?.areValuesEqual && (
              <span className="text-gray-400"> (Same as Quarter to Date)</span>
            )}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

  const CustomLegend = ({ payload }: any) => {
    return (
      <ul className="flex flex-wrap justify-center gap-4 mt-4">
        {payload.map((entry: any, index: number) => (
          <li key={`item-${index}`} className="flex items-center">
            <div
              className="w-3 h-3 rounded-full mr-2"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-sm text-gray-600">{entry.value}</span>
          </li>
        ))}
      </ul>
    );
  };

  const CustomXAxisTick = ({ x, y, payload }: any) => {
    const maxWidth = 15;

    const wrapText = (text: string) => {
      const words = text.split(" ");
      let line = "";
      const lines: string[] = [];

      words.forEach((word) => {
        if ((line + word).length > maxWidth) {
          lines.push(line.trim());
          line = word + " ";
        } else {
          line += word + " ";
        }
      });

      if (line) lines.push(line.trim());
      return lines;
    };

    const wrappedText = wrapText(payload.value);

    return (
      <g transform={`translate(${x},${y})`}>
        {wrappedText.map((line: string, index: number) => (
          <text
            key={index}
            x={0}
            y={index * 12}
            textAnchor="end" // Change from "middle" to "end"
            fill="#4b5563"
            fontSize={10}
            dy={10}
            transform="rotate(-45)" // Add this transform
          >
            {line}
          </text>
        ))}
      </g>
    );
  };

  if (loading)
    return (
      <div className="flex justify-center items-center h-full p-6">
        <Loader />
      </div>
    );
  if (error)
    return (
      <div className="flex justify-center items-center h-full p-6 text-red-500">
        <CircleAlert className="mr-2" />
        <span className="font-medium">Error loading data. Please try again.</span>
      </div>
    );
  if (noData || chartData.length === 0)
    return (
      <div className="h-full w-full bg-[#FFFFFF] rounded-md shadow hover:shadow-lg border border-[#E0E5EF] p-4 flex flex-col transition-all duration-300">
        <h2 className="text-lg font-bold text-[#233E7D] tracking-tight">{title}</h2>
        <div className="flex justify-center items-center h-full p-4 text-[#6B7280]">
          <span>Data Insufficient for Chart Visualization</span>
        </div>
      </div>
    );

  return (
    <div className="h-full w-full bg-[#FFFFFF] rounded-md shadow hover:shadow-lg border border-[#E0E5EF] flex flex-col p-6 transition-all duration-300">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-bold text-[#233E7D] tracking-tight">{title}</h2>
        <BarChartIcon className="w-6 h-6 text-[#D82026]" />
      </div>
      <div className="flex-grow">
        <ResponsiveContainer width="100%" height={500}>
          <BarChart 
            data={chartData}
            margin={{ top: 20, right: 30, left: 65, bottom: 90 }}
          >   
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E5EF" />
            <XAxis 
              dataKey="name" 
              tick={<CustomXAxisTick />} 
              tickLine={false} 
              interval={0}
              height={100}
              angle={-45}
            />
            <YAxis
              tickFormatter={(value) => `$${value.toFixed(2)}`}
              tick={{ fill: "#233E7D", fontWeight: 500, fontSize: 12 }}
              domain={[0, "auto"]}
              minTickGap={1}
            />
            <Tooltip content={<CustomTooltip />} wrapperStyle={{ background: '#fff', border: '1px solid #E0E5EF', borderRadius: '0.375rem', color: '#233E7D' }} labelStyle={{ color: '#233E7D', fontWeight: 600 }} itemStyle={{ color: '#233E7D' }} />
            <Legend content={<CustomLegend />} wrapperStyle={{ color: '#233E7D', fontWeight: 500, fontSize: 12 }} />
            <Bar dataKey="Month to Date" stackId="a" fill={COLORS[0]} />
            {chartData.some((item) => item.showQuarterToDate) && (
              <Bar dataKey="Quarter to Date" stackId="a" fill={COLORS[1]} />
            )}
            <Bar dataKey="Year to Date" stackId="a" fill={COLORS[2]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
    </div>
  );
};

export default StackedBarChart;