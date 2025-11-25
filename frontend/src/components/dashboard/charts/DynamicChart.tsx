import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
  ComposedChart,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  RadialBarChart,
  RadialBar,
  FunnelChart,
  Funnel,
  LabelList,
  Treemap,
  Sankey,
} from "recharts";

// Define the possible chart types
type ChartType =
  | "bar"
  | "line"
  | "pie"
  | "area"
  | "scatter"
  | "composed"
  | "radar"
  | "radialBar"
  | "funnel"
  | "treemap"
  | "sankey";

// Define the props interface
interface DynamicChartProps {
  data: Array<Record<string, any>>;
  type: ChartType;
  xKey: string;
  yKey: string;
  name: string;
  layout?: "vertical" | "horizontal";
  height?: number;
  xAxisLabelFormatter?: (label: string) => string;
  xAxisTooltipFormatter?: (label: string) => string;
  legendTextColor?: string;
}

const truncateLegendText = (text: string, maxLength: number = 20) => {
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
};

const COLORS: string[] = [
  "#233E7D", // Brand Blue
  "#D82026", // Brand Red
  "#3b82f6", // Accent Blue
  "#22C55E", // Success Green
  "#F59E0B", // Yellow
  "#8b5cf6", // Purple
  "#06b6d4", // Cyan
  "#f97316", // Orange
  "#E0E5EF", // Brand Border Gray
  "#6B7280", // Text Gray
];

const DynamicChart: React.FC<DynamicChartProps> = ({
  data,
  type,
  xKey,
  yKey,
  name,
  layout = "vertical",
  height = 300,
  xAxisLabelFormatter,
  xAxisTooltipFormatter,
  legendTextColor = "black",
}) => {
  const renderChart = () => {
    switch (type) {
      case "bar":
        return (
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E5EF" />
            <XAxis
              dataKey={xKey}
              tickFormatter={xAxisLabelFormatter}
              interval={0}
              tick={{ fontSize: 12, fill: '#6B7280' }}
              axisLine={{ stroke: '#E0E5EF' }}
              tickLine={{ stroke: '#E0E5EF' }}
            />
            <YAxis tick={{ fontSize: 12, fill: '#6B7280' }} axisLine={{ stroke: '#E0E5EF' }} tickLine={{ stroke: '#E0E5EF' }} />
            <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #E0E5EF', color: '#233E7D', fontWeight: 500 }} />
            <Legend wrapperStyle={{ color: '#233E7D', fontSize: '12px' }} />
            <Bar dataKey={yKey} fill="#233E7D" name={name} />
          </BarChart>
        );
      case "line":
        return (
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E5EF" />
            <XAxis
              dataKey={xKey}
              tickFormatter={xAxisLabelFormatter}
              interval={0}
              height={100}
              angle={-90}
              textAnchor="end"
              tick={{ fontSize: 12, fill: '#6B7280' }}
              axisLine={{ stroke: '#E0E5EF' }}
              tickLine={{ stroke: '#E0E5EF' }}
            />
            <YAxis tick={{ fontSize: 12, fill: '#6B7280' }} axisLine={{ stroke: '#E0E5EF' }} tickLine={{ stroke: '#E0E5EF' }} />
            <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #E0E5EF', color: '#233E7D', fontWeight: 500 }} />
            <Legend wrapperStyle={{ color: '#233E7D', fontSize: '12px' }} />
            <Line type="monotone" dataKey={yKey} stroke="#233E7D" name={name} />
          </LineChart>
        );
      case "pie":
        return (
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius="80%"
              fill="#8884d8"
              dataKey={yKey}
              nameKey={xKey}
            >
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                />
              ))}
            </Pie>
            <Tooltip formatter={(value) => `$${Number(value).toFixed(1)}`} />
            <Legend
              layout="vertical"
              align="right"
              verticalAlign="middle"
              formatter={(value) => truncateLegendText(value)}
              payload={data.map((item, index) => ({
                value: truncateLegendText(item[xKey]),
                type: "square",
                color: COLORS[index % COLORS.length],
                id: item[xKey],
              }))}
              wrapperStyle={{ color: legendTextColor }}
            />
          </PieChart>
        );
      case "area":
        return (
          <AreaChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E5EF" />
            <XAxis dataKey={xKey} tickFormatter={xAxisLabelFormatter} tick={{ fontSize: 12, fill: '#6B7280' }} axisLine={{ stroke: '#E0E5EF' }} tickLine={{ stroke: '#E0E5EF' }} />
            <YAxis tick={{ fontSize: 12, fill: '#6B7280' }} axisLine={{ stroke: '#E0E5EF' }} tickLine={{ stroke: '#E0E5EF' }} />
            <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #E0E5EF', color: '#233E7D', fontWeight: 500 }} />
            <Legend wrapperStyle={{ color: '#233E7D', fontSize: '12px' }} />
            <Area
              type="monotone"
              dataKey={yKey}
              stroke="#233E7D"
              fill="#233E7D"
              name={name}
            />
          </AreaChart>
        );
      case "scatter":
        return (
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E5EF" />
            <XAxis dataKey={xKey} type="number" name={xKey} />
            <YAxis dataKey={yKey} type="number" name={yKey} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ backgroundColor: '#fff', border: '1px solid #E0E5EF', color: '#233E7D', fontWeight: 500 }} />
            <Legend wrapperStyle={{ color: '#233E7D', fontSize: '12px' }} />
            <Scatter name={name} data={data} fill="#8884d8" />
          </ScatterChart>
        );
      case "composed":
        return (
          <ComposedChart data={data}>
            <CartesianGrid stroke="#f5f5f5" />
            <XAxis dataKey={xKey} tickFormatter={xAxisLabelFormatter} />
            <YAxis />
            <Tooltip formatter={xAxisTooltipFormatter} />
            <Legend />
            <Area
              type="monotone"
              dataKey={yKey}
              fill="#233E7D"
              stroke="#233E7D"
            />
            <Bar dataKey={yKey} barSize={20} fill="#413ea0" />
            <Line type="monotone" dataKey={yKey} stroke="#ff7300" />
          </ComposedChart>
        );
      case "radar":
        return (
          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
            <PolarGrid />
            <PolarAngleAxis dataKey={xKey} />
            <PolarRadiusAxis />
            <Radar
              name={name}
              dataKey={yKey}
              stroke="#8884d8"
              fill="#8884d8"
              fillOpacity={0.6}
            />
          </RadarChart>
        );
      case "radialBar":
        return (
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="10%"
            outerRadius="80%"
            barSize={10}
            data={data}
          >
            <RadialBar
              label={{ position: "insideStart", fill: "#fff" }}
              background
              dataKey={yKey}
            />
            <Legend
              iconSize={10}
              layout="vertical"
              verticalAlign="middle"
              align="right"
            />
          </RadialBarChart>
        );
      case "funnel":
        return (
          <FunnelChart>
            <Tooltip />
            <Funnel dataKey={yKey} data={data} isAnimationActive>
              <LabelList
                position="right"
                fill="#000"
                stroke="none"
                dataKey={xKey}
              />
            </Funnel>
          </FunnelChart>
        );
      case "treemap":
        return (
            <Treemap
              data={data}
              dataKey={yKey}
              aspectRatio={4 / 3}
              stroke="#fff"
              fill="#233E7D"
            >
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #E0E5EF', color: '#233E7D', fontWeight: 500 }} />
            </Treemap>
        );
      case "sankey":
        const sankeyData = {
          nodes: data.map((item) => ({ name: item[xKey] })),
          links: data.map((item) => ({
            source: item[xKey],
            target: item[yKey],
            value: item.value,
          })),
        };
        return (
            <Sankey
              data={sankeyData}
              node={{ fill: "#233E7D", stroke: "#fff" }}
              link={{ stroke: "#22C55E" }}
            >
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #E0E5EF', color: '#233E7D', fontWeight: 500 }} />
            </Sankey>
        );
      default:
        return <div>Unsupported chart type</div>;
    }
  };

  return (
    <div style={{ width: "100%", height: "100%" }}>
      <ResponsiveContainer>{renderChart()}</ResponsiveContainer>
    </div>
  );
};

export default DynamicChart;
