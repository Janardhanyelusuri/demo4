import React, { useMemo } from "react";
import { BarChart3 } from "lucide-react";
import {
  BarChart,
  Bar,
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

interface UtilizationBarChartProps {
  data: UtilizationDataPoint[];
  title: string;
  groupBy?: "instance_type" | "resourceregion" | "resource_group"; // What to group by
  metricName?: string; // Filter by specific metric
  topN?: number; // Show top N items
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
const COLORS = [
  BRAND_BLUE, BRAND_RED, '#22C55E', '#FFC658', '#8884d8', '#FF7300', '#00CFFF', '#B81A1F', '#19294e', '#C8C8C8'
];

const UtilizationBarChart: React.FC<UtilizationBarChartProps> = ({
  data,
  title,
  groupBy = "instance_type",
  metricName = "Percentage CPU",
  topN = 10,
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
          values: [],
          vms: new Set(),
        };
      }
      acc[groupKey].values.push(item.value);
      acc[groupKey].vms.add(item.vm_name);
      return acc;
    }, {} as any);

    // Convert to chart format with statistics
    const formattedData = Object.values(groupedData).map((group: any) => {
      const values = group.values.sort((a: number, b: number) => a - b);
      const avg = values.reduce((sum: number, val: number) => sum + val, 0) / values.length;
      const min = values[0];
      const max = values[values.length - 1];
      const unitText = filteredData[0]?.unit || "";
      const formattedUnit = unitText === 'Percent' ? '%' : unitText;
      
      return {
        name: group.name,
        average: parseFloat(avg.toFixed(2)),
        minimum: parseFloat(min.toFixed(2)),
        maximum: parseFloat(max.toFixed(2)),
        vmCount: group.vms.size,
        unit: formattedUnit,
      };
    });

    // Sort by average value (descending) and take top N
    formattedData.sort((a, b) => b.average - a.average);
    
    return formattedData.slice(0, topN);
  }, [data, groupBy, metricName, topN]);

  if (chartData.length === 0) {
    return (
      <div className="bg-[#fff] p-6 rounded-md shadow-md border border-[#E0E5EF] h-96 flex flex-col items-center justify-center">
        <BarChart3 className="w-12 h-12 text-[#C8C8C8] mb-4" />
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
      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={BRAND_BORDER} />
            <XAxis 
              dataKey="name" 
              tick={{ fontSize: 10, fill: BRAND_TEXT_SECONDARY }}
              angle={-45}
              textAnchor="end"
              height={60}
              interval={0}
              tickFormatter={(value) => {
                // Truncate long names on X-axis
                return value.length > 10 ? `${value.slice(0, 10)}...` : value;
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
                name === 'average' ? 'Average' : 
                name === 'minimum' ? 'Minimum' :
                name === 'maximum' ? 'Maximum' : name
              ]}
              labelFormatter={(label) => `${groupBy.replace('_', ' ')}: ${label}`}
              contentStyle={{
                backgroundColor: BRAND_CARD,
                border: `1px solid ${BRAND_BORDER}`,
                borderRadius: '8px',
                fontSize: '12px',
                color: BRAND_TEXT,
                boxShadow: '0 4px 6px rgba(35, 62, 125, 0.05)',
                padding: '8px 12px'
              }}
              cursor={{ fill: BRAND_BLUE + '10' }}
            />
            <Legend 
              verticalAlign="top" 
              height={36}
              wrapperStyle={{ 
                fontSize: '12px',
                color: BRAND_TEXT,
                paddingBottom: '10px'
              }}
            />
            <Bar dataKey="average" fill={BRAND_BLUE} name="Average" radius={[6, 6, 0, 0]} />
            <Bar dataKey="minimum" fill={BRAND_RED} name="Minimum" radius={[6, 6, 0, 0]} />
            <Bar dataKey="maximum" fill="#22C55E" name="Maximum" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
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
                {item.average} {item.unit} <span className="text-xs text-[#C8C8C8]">({item.vmCount} VMs)</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default UtilizationBarChart;
