// "use client";
// import React, { useEffect, useState } from "react";
// import {
//   LineChart,
//   Line,
//   XAxis,
//   YAxis,
//   CartesianGrid,
//   Tooltip,
//   Legend,
//   ResponsiveContainer,
// } from "recharts";
// import { fetchdDataDashboard } from "@/lib/api";
// import Loader from "@/components/Loader";
// import { CircleAlert, LineChart as LineChartIcon } from "lucide-react";
// import apiQueue from "@/lib/apiQueueManager";

// type DataItem = {
//   service_name: string;
//   costs: {
//     month_to_date: number | null;
//     quarter_to_date: string;
//     year_to_date: string;
//   };
// };

// type Props = {
//   cloudPlatform?: string;
//   queryType?: string;
//   id?: string;
//   title: string;
// };

// const COLORS = [
//   "#3b82f6",
//   "#10b981",
//   "#f59e0b",
//   "#ef4444",
//   "#8b5cf6",
//   "#ec4899",
//   "#14b8a6",
//   "#f97316",
//   "#6366f1",
//   "#84cc16",
//   "#06b6d4",
//   "#d946ef",
// ];
// // 
// const CostChart: React.FC<Props> = ({
//   cloudPlatform,
//   queryType,
//   id,
//   title,
// }) => {
//   const [data, setData] = useState<DataItem[]>([]);
//   const [loading, setLoading] = useState<boolean>(true);
//   const [error, setError] = useState<Error | null>(null);

//   useEffect(() => {
//     const getData = async () => {
//       let retries = 0;
//       const maxRetries = 3;

//       while (retries < maxRetries) {
//         try {
//           if (cloudPlatform && queryType && id) {
//             const result = await apiQueue.enqueue(
//               () => fetchdDataDashboard(
//               cloudPlatform,
//               queryType,
//               id.toString()
//             ),"CostChart");
//             setData(result);
//             setLoading(false);
//             break;
//           } else {
//             throw new Error("Missing required parameters");
//           }
//         } catch (error) {
//           retries++;
//           if (retries === maxRetries) {
//             setError(
//               error instanceof Error
//                 ? error
//                 : new Error("An unknown error occurred")
//             );
//             setLoading(false);
//           }
//         }
//       }
//     };

//     getData();
//   }, [cloudPlatform, queryType, id]);

//   const chartData = data.map((item) => ({
//     name: item.service_name,
//     "Month to Date":
//       item.costs.month_to_date !== null
//         ? parseFloat(item.costs.month_to_date.toString())
//         : 0,
//     "Quarter to Date": parseFloat(item.costs.quarter_to_date),
//     "Year to Date": parseFloat(item.costs.year_to_date),
//   }));

//   const CustomTooltip = ({ active, payload, label }: any) => {
//     if (active && payload && payload.length) {
//       return (
//         <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
//           <p className="font-semibold text-gray-800">{label}</p>
//           {payload.map((entry: any, index: number) => (
//             <p key={index} className="text-gray-600">
//               {`${entry.name}: $${entry.value.toFixed(1)}`}
//             </p>
//           ))}
//         </div>
//       );
//     }
//     return null;
//   };

//   const CustomLegend = ({ payload }: any) => {
//     return (
//       <ul className="flex flex-wrap justify-center gap-4 mt-4">
//         {payload.map((entry: any, index: number) => (
//           <li key={`item-${index}`} className="flex items-center">
//             <div
//               className="w-3 h-3 rounded-full mr-2"
//               style={{ backgroundColor: entry.color }}
//             />
//             <span className="text-sm text-gray-600">{entry.value}</span>
//           </li>
//         ))}
//       </ul>
//     );
//   };

//   if (loading)
//     return (
//       <div className="flex justify-center items-center h-full p-6">
//         <Loader />
//       </div>
//     );
//   if (error)
//     return (
//       <div className="flex justify-center items-center h-full p-6 text-red-500">
//         <CircleAlert className="mr-2" />
//         <span className="font-medium">
//           Error loading data. Please try again.
//         </span>
//       </div>
//     );

//   return (
//     <div className="h-full w-full bg-white rounded-lg shadow-lg border border-gray-200 flex flex-col p-6">
//       <div className="flex items-center justify-between mb-6">
//         <h2 className="text-lg font-semibold text-gray-700">{title}</h2>
//         <LineChartIcon className="w-6 h-6 text-blue-500" />
//       </div>
//       <div className="flex-grow">
//         <ResponsiveContainer width="100%" height="100%">
//           <LineChart data={chartData}>
//             <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
//             <XAxis dataKey="name" tick={{ fill: "#4b5563" }} />
//             <YAxis
//               tickFormatter={(value) => `$${value.toFixed(1)}`}
//               tick={{ fill: "#4b5563" }}
//             />
//             <Tooltip content={<CustomTooltip />} />
//             <Legend content={<CustomLegend />} />
//             <Line
//               type="monotone"
//               dataKey="Month to Date"
//               stroke={COLORS[0]}
//               strokeWidth={2}
//               dot={false}
//               activeDot={{ r: 5 }}
//             />
//             <Line
//               type="monotone"
//               dataKey="Quarter to Date"
//               stroke={COLORS[1]}
//               strokeWidth={2}
//               dot={false}
//               activeDot={{ r: 5 }}
//             />
//             <Line
//               type="monotone"
//               dataKey="Year to Date"
//               stroke={COLORS[2]}
//               strokeWidth={2}
//               dot={false}
//               activeDot={{ r: 5 }}
//             />
//           </LineChart>
//         </ResponsiveContainer>
//       </div>
//     </div>
//   );
// };

// export default CostChart;
