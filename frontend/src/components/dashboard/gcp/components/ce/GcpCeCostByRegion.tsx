import React, { useEffect, useState } from "react";
import DynamicChart from "../../../charts/DynamicChart";
import { fetchData } from "@/lib/api";
import Loader from "@/components/Loader";
type Props = {
  projectId: string;
};

interface CostItem {
  "gcp_location_dim.region_name": string;
  "gcp_fact_dim.total_list_cost": string; // Change type to string
}

const GcpCeCostByRegion: React.FC<Props> = ({ projectId }) => {
  const [response, setResponse] = useState<CostItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const cloudPlatform = "gcp";
  const queryType = "gcp_ce_cost_by_region";
  const topItemsToShow = 6; // Reduced to show top 5 plus "Others"

  useEffect(() => {
    const getData = async () => {
      if (!cloudPlatform) return;

      try {
        const result = await fetchData(cloudPlatform, queryType, projectId);
        setResponse(result as CostItem[]);
      } catch (error) {
        setError(
          error instanceof Error
            ? error
            : new Error("An unknown error occurred")
        );
      } finally {
        setLoading(false);
      }
    };

    getData();
  }, [cloudPlatform]);

  if (loading)
    return (
      <div>
        <Loader />
      </div>
    );
  if (error) return <div>Error: {error.message}</div>;

  // Format and process the data for the chart
  const sortedData = response
    .map((item) => ({
      name: item["gcp_location_dim.region_name"],
      value: parseFloat(item["gcp_fact_dim.total_list_cost"]), // Convert to number
    }))
    .filter((item) => item.value > 0)
    .sort((a, b) => b.value - a.value);

  const topItems = sortedData.slice(0, topItemsToShow - 1);
  const otherItems = sortedData.slice(topItemsToShow - 1);

  const chartData = [
    ...topItems,
    {
      name: "Others",
      value: otherItems.reduce((sum, item) => sum + item.value, 0),
    },
  ];

  // Calculate total cost
  const totalCost = chartData.reduce((sum, item) => sum + item.value, 0);

  // Add percentage to the name
  const formattedChartData = chartData.map((item) => ({
    ...item,
    name: `${item.name} (${((item.value / totalCost) * 100).toFixed(1)}%)`,
  }));

  return (
    <div className="h-full w-full bg-white rounded-lg shadow-md p-4 border flex flex-col">
      <h2 className="mb-4 text-lg">Cost by Region</h2>
      <div className="flex-grow">
        <DynamicChart
          data={formattedChartData}
          type="pie"
          xKey="name"
          yKey="value"
          name="Cost ($)"
        />
      </div>
      {/* <div className="mt-4 text-center text-sm text-gray-600">
        Total Cost: ${totalCost.toFixed(1)}
      </div> */}
    </div>
  );
};

export default GcpCeCostByRegion;
