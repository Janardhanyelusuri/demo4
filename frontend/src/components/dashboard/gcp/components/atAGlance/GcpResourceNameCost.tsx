import React, { useState, useEffect } from "react";
import { fetchData } from "@/lib/api";
import DynamicChart from "../../../charts/DynamicChart";
import Loader from "@/components/Loader";
type Props = {
  projectId: string;
};

const GcpResourceNameCost: React.FC<Props> = ({ projectId }) => {
  const [response, setResponse] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const cloudPlatform = "gcp";
  const queryType = "gcp_resource_name_cost";

  useEffect(() => {
    const getData = async () => {
      if (!cloudPlatform) return;

      try {
        const result = await fetchData(cloudPlatform, queryType, projectId);
        setResponse(result);
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

  const prepareChartData = (data: any) => {
    const filteredData = data
      .filter(
        (item: any) => parseFloat(item["gcp_fact_dim.total_billed_cost"]) > 0
      )
      .map((item: any) => ({
        name: item["gcp_fact_dim.resource_name"],
        value: parseFloat(item["gcp_fact_dim.total_billed_cost"]),
      }))
      .sort((a: any, b: any) => b.value - a.value)
      .slice(0, 10); // Get top 10 resources by cost

    // If no data remains after filtering, add a placeholder entry
    if (filteredData.length === 0) {
      return [{ name: "No Cost Data Available", value: 1 }];
    }

    return filteredData;
  };

  if (loading)
    return (
      <div>
        <Loader />
      </div>
    );
  if (error) return <div>Error: {error.message}</div>;

  const chartData = prepareChartData(response);

  return (
    <div className="h-full w-full bg-white rounded-lg shadow-md p-4 border flex flex-col">
      <h2 className="mb-4">GCP Resource Costs</h2>
      <div className="flex-grow mt-4">
        <DynamicChart
          data={chartData}
          type="pie"
          xKey="name"
          yKey="value"
          name="Cost Distribution"
          height={300}
        />
      </div>
    </div>
  );
};

export default GcpResourceNameCost;
