import React, { useMemo } from "react";
import { LineChart as LineChartIcon } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface UtilizationDataPoint {
  vm_name: string;
  resource_group: string;
  subscription_id: string;
  timestamp: string;
  value: number;
  metric_name: string;
  unit: string;
  displaydescription: string;
  namespace: string;
  resourceregion: string;
  resource_id: string;
  instance_type: string;
}

interface UtilizationLineChartProps {
  data: UtilizationDataPoint[];
  title: string;
  metricName?: string; // Filter by specific metric
  vmName?: string; // Filter by specific VM
}

// CloudPulse Brand Colors
const BRAND_BLUE = '#233E7D';
const BRAND_RED = '#D82026';
const BRAND_BG = '#F9FEFF';
const BRAND_CARD = '#fff';
const BRAND_BORDER = '#E0E5EF';
const BRAND_TEXT = '#233E7D';
const BRAND_TEXT_SECONDARY = '#6B7280';
const BRAND_GRAY = '#C8C8C8';

const UtilizationLineChart: React.FC<UtilizationLineChartProps> = ({
  data,
  title,
  metricName = "Percentage CPU",
  vmName,
}) => {
  const chartData = useMemo(() => {
    // Filter by metric name and VM if specified
    let filteredData = data.filter(item => item.metric_name === metricName);
    if (vmName) {
      filteredData = filteredData.filter(item => item.vm_name === vmName);
    }

    if (filteredData.length === 0) return [];

    // Group data by timestamp and aggregate values
    const groupedData = filteredData.reduce((acc, item) => {
      // Keep the original timestamp for sorting, but use formatted date for display
      const originalDate = new Date(item.timestamp);
      const formattedDate = originalDate.toLocaleDateString();
      
      if (!acc[formattedDate]) {
        acc[formattedDate] = {
          timestamp: formattedDate,
          originalTimestamp: originalDate, // Keep original for proper sorting
          totalValue: 0,
          count: 0,
          vms: new Set(),
        };
      }
      acc[formattedDate].totalValue += item.value;
      acc[formattedDate].count += 1;
      acc[formattedDate].vms.add(item.vm_name);
      return acc;
    }, {} as any);

    // Convert to chart format
    const formattedData = Object.values(groupedData).map((item: any) => {
      const unitText = filteredData[0]?.unit || "";
      const formattedUnit = unitText === 'Percent' ? '%' : unitText;
      
      return {
        timestamp: item.timestamp,
        originalTimestamp: item.originalTimestamp,
        averageValue: parseFloat((item.totalValue / item.count).toFixed(2)),
        vmCount: item.vms.size,
        unit: formattedUnit,
      };
    });

    // Sort by original timestamp (ascending order - oldest to newest)
    formattedData.sort((a, b) => a.originalTimestamp.getTime() - b.originalTimestamp.getTime());

    return formattedData;
  }, [data, metricName, vmName]);

  if (chartData.length === 0) {
    return (
      <div className="bg-[#fff] p-6 rounded-md shadow-md border border-[#E0E5EF] h-96 flex flex-col items-center justify-center">
        <LineChartIcon className="w-12 h-12 text-[#C8C8C8] mb-4" />
        <h3 className="text-lg font-semibold text-[#233E7D] mb-2">{title}</h3>
        <p className="text-[#6B7280] text-center">
          No utilization data available for {metricName}
        </p>
      </div>
    );
  }

  return (
    <div className="bg-[#fff] p-6 rounded-md shadow-md border border-[#E0E5EF]">
      <h3 className="text-lg font-semibold text-[#233E7D] mb-4">{title}</h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E5EF" />
            <XAxis 
              dataKey="timestamp" 
              tick={{ fontSize: 10, fill: BRAND_TEXT_SECONDARY }}
              angle={-45}
              textAnchor="end"
              height={80}
              tickFormatter={(value) => {
                // Truncate long dates
                return value.length > 8 ? `${value.slice(0, 8)}...` : value;
              }}
              axisLine={{ stroke: BRAND_BORDER }}
              tickLine={{ stroke: BRAND_BORDER }}
            />
            <YAxis 
              tick={{ fontSize: 12, fill: BRAND_TEXT_SECONDARY }}
              axisLine={{ stroke: BRAND_BORDER }}
              tickLine={{ stroke: BRAND_BORDER }}
            />
            <Tooltip 
              formatter={(value: any, name: string) => [
                `${value} ${chartData[0]?.unit || ''}`, 
                name === 'averageValue' ? `Average ${metricName}` : name
              ]}
              labelFormatter={(label) => `Date: ${label}`}
              contentStyle={{
                backgroundColor: BRAND_CARD,
                border: `1px solid ${BRAND_BORDER}`,
                borderRadius: '8px',
                fontSize: '12px',
                color: BRAND_TEXT
              }}
              labelStyle={{ color: BRAND_TEXT_SECONDARY }}
              itemStyle={{ color: BRAND_TEXT }}
            />
            <Legend 
              wrapperStyle={{ fontSize: '12px', color: BRAND_TEXT }}
              iconType="circle"
              formatter={(value) => {
                // Truncate legend text for better display
                return value.length > 20 ? `${value.slice(0, 20)}...` : value;
              }}
            />
            <Line
              type="monotone"
              dataKey="averageValue"
              stroke={BRAND_BLUE}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6, fill: BRAND_RED, stroke: BRAND_BLUE, strokeWidth: 2 }}
              name={`Average ${metricName}`}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Summary - Below Chart */}
      <div className="mt-4 border-t border-[#E0E5EF] pt-4">
        <h4 className="font-semibold text-lg text-[#233E7D] mb-3">Summary</h4>
        <div className="space-y-3">
          <div className="bg-[#F9FEFF] p-4 rounded-md border border-[#E0E5EF] flex items-center justify-between">
            <span className="text-base text-[#233E7D] font-medium">Total Data Points:</span>
            <span className="text-xl font-bold text-[#233E7D]">{chartData.length}</span>
          </div>
          <div className="bg-[#F9FEFF] p-4 rounded-md border border-[#E0E5EF] flex items-center justify-between">
            <span className="text-base text-[#233E7D] font-medium">Average {metricName}:</span>
            <span className="text-xl font-bold text-[#233E7D]">
              {chartData.length > 0 ? 
                `${(chartData.reduce((sum, item) => sum + item.averageValue, 0) / chartData.length).toFixed(2)} ${chartData[0]?.unit || ''}` 
                : 'N/A'
              }
            </span>
          </div>
          <div className="bg-[#F9FEFF] p-4 rounded-md border border-[#E0E5EF] flex items-center justify-between">
            <span className="text-base text-[#233E7D] font-medium">Peak Value:</span>
            <span className="text-xl font-bold text-[#233E7D]">
              {chartData.length > 0 ? 
                `${Math.max(...chartData.map(item => item.averageValue)).toFixed(2)} ${chartData[0]?.unit || ''}` 
                : 'N/A'
              }
            </span>
          </div>
          <div className="bg-[#F9FEFF] p-4 rounded-md border border-[#E0E5EF] flex items-center justify-between">
            <span className="text-base text-[#233E7D] font-medium">Date Range:</span>
            <span className="text-xl font-bold text-[#233E7D]">
              {chartData.length > 0 ? 
                `${chartData[0]?.timestamp} - ${chartData[chartData.length - 1]?.timestamp}` 
                : 'N/A'
              }
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UtilizationLineChart;
