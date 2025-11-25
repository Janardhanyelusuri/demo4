import React, { useMemo } from "react";
import { PieChart as PieChartIcon } from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
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

interface UtilizationPieChartProps {
  data: UtilizationDataPoint[];
  title: string;
  groupBy?: "instance_type" | "resourceregion" | "resource_group"; // What to group by
  metricName?: string; // Filter by specific metric
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
// Pie chart color palette (primary blue, red, green, yellow, purple, etc.)
const COLORS = [
  BRAND_BLUE, BRAND_RED, '#22C55E', '#FFC658', '#8884d8', '#FF7300', '#00CFFF', '#B81A1F', '#19294e', '#C8C8C8'
];

const UtilizationPieChart: React.FC<UtilizationPieChartProps> = ({
  data,
  title,
  groupBy = "instance_type",
  metricName = "Percentage CPU",
}) => {
  const chartData = useMemo(() => {
    // Filter by metric name
    const filteredData = data.filter(item => item.metric_name === metricName);

    if (filteredData.length === 0) return [];

    // Group data by the specified field and calculate statistics
    const groupedData = filteredData.reduce((acc, item) => {
      const groupKey = item[groupBy] || "Unknown";
      if (!acc[groupKey]) {
        acc[groupKey] = {
          name: groupKey,
          totalValue: 0,
          count: 0,
          vms: new Set(),
        };
      }
      acc[groupKey].totalValue += item.value;
      acc[groupKey].count += 1;
      acc[groupKey].vms.add(item.vm_name);
      return acc;
    }, {} as any);

    // Convert to chart format
    const formattedData = Object.values(groupedData).map((group: any) => {
      const unitText = filteredData[0]?.unit || "";
      const formattedUnit = unitText === 'Percent' ? '%' : unitText;
      
      return {
        name: group.name,
        value: parseFloat((group.totalValue / group.count).toFixed(2)),
        vmCount: group.vms.size,
        unit: formattedUnit,
      };
    });

    // Sort by value (descending)
    formattedData.sort((a, b) => b.value - a.value);

    return formattedData;
  }, [data, groupBy, metricName]);

  if (chartData.length === 0) {
    return (
      <div className="bg-[#fff] p-6 rounded-md shadow-md border border-[#E0E5EF] h-96 flex flex-col items-center justify-center">
        <PieChartIcon className="w-12 h-12 text-[#C8C8C8] mb-4" />
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
    {/* Pie Chart - Centered */}
    <div className="h-80 mb-4">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={90}
            innerRadius={50}
            paddingAngle={2}
            labelLine={false}
            label={({ name, percent }) => percent > 0.05 ? `${name}` : ''}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke={BRAND_BORDER} strokeWidth={2} />
            ))}
          </Pie>
          <Tooltip
            formatter={(value: any, name: string, props: any) => [
              `${value} ${chartData[0]?.unit || ''}`,
              name
            ]}
            contentStyle={{
              backgroundColor: BRAND_CARD,
              border: `1px solid ${BRAND_BORDER}`,
              borderRadius: '8px',
              fontSize: '12px',
              color: BRAND_TEXT,
              boxShadow: '0 4px 6px rgba(35, 62, 125, 0.05)',
              padding: '8px 12px'
            }}
            labelStyle={{ color: BRAND_TEXT_SECONDARY }}
            itemStyle={{ color: BRAND_TEXT }}
          />
          <Legend
            verticalAlign="bottom"
            height={36}
            wrapperStyle={{
              fontSize: '12px',
              color: BRAND_TEXT,
              paddingTop: '10px',
            }}
            iconType="circle"
            formatter={(value) => value.length > 20 ? `${value.slice(0, 20)}...` : value}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
    {/* Summary - Below Chart */}
    <div className="mt-4 border-t border-[#E0E5EF] pt-4">
      <h4 className="font-semibold text-lg text-[#233E7D] mb-3">Summary</h4>
      <div className="space-y-2">
        {chartData.slice(0, 5).map((item, index) => (
          <div key={item.name} className="flex items-center justify-between p-4 bg-[#F9FEFF] rounded-md border border-[#E0E5EF] hover:bg-[#233E7D]/5 hover:shadow-md transition-all duration-200 group">
            <div className="flex items-center min-w-0 flex-1">
              <div
                className="w-4 h-4 rounded-full mr-3 flex-shrink-0 group-hover:scale-110 transition-transform duration-200"
                style={{ backgroundColor: COLORS[index % COLORS.length] }}
              />
              <span
                className="text-base font-semibold text-[#233E7D] truncate group-hover:text-[#D82026]"
                title={item.name}
              >
                {item.name}
              </span>
            </div>
            <div className="text-[#6B7280] text-base ml-3 flex-shrink-0 font-medium group-hover:text-[#233E7D]">
              {item.value} {item.unit} <span className="text-xs text-[#C8C8C8]">({item.vmCount} VMs)</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  </div>
);
// Removed duplicate/old JSX block at the end of the file. Only the main return block above is used.
};

export default UtilizationPieChart;
