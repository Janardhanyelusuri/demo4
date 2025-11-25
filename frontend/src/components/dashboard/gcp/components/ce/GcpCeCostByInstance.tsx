import React, { useEffect, useState } from "react";
import DynamicChart from "../../../charts/DynamicChart";
import { fetchData } from "@/lib/api";
import Loader from "@/components/Loader";
type Props = {
  projectId: string;
};
interface CostItem {
  "gcp_fact_dim.resource_name": string;
  "gcp_fact_dim.total_list_cost": string;
}

const GcpCeCostByInstance: React.FC<Props> = ({ projectId }) => {
  const [response, setResponse] = useState<CostItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const cloudPlatform = "gcp";
  const queryType = "gcp_ce_cost_by_instance";
  const topItemsToShow = 6;

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

  const sortedData = response
    .map((item) => ({
      name: item["gcp_fact_dim.resource_name"],
      value: parseFloat(item["gcp_fact_dim.total_list_cost"]),
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

  const totalCost = chartData.reduce((sum, item) => sum + item.value, 0);

  const shortenName = (name: string) => {
    const parts = name.split("/");
    if (parts.length > 1) {
      return parts[1].slice(0, 4) + "...";
    } else {
      return name.slice(0, 4) + "...";
    }
  };

  const formattedChartData = chartData.map((item) => ({
    ...item,
    name:
      item.name === "Others"
        ? `Others (${((item.value / totalCost) * 100).toFixed(1)}%)`
        : `${shortenName(item.name)} (${(
            (item.value / totalCost) *
            100
          ).toFixed(1)}%)`,
  }));

  return (
    <div className="h-full w-full bg-white rounded-lg shadow-md p-4 border flex flex-col">
      <h2 className="mb-4 text-lg">Cost By Instance</h2>
      <div className="flex-grow">
        <DynamicChart
          data={formattedChartData}
          type="pie"
          xKey="name"
          yKey="value"
          name="Cost ($)"
        />
      </div>
    </div>
  );
};

export default GcpCeCostByInstance;
