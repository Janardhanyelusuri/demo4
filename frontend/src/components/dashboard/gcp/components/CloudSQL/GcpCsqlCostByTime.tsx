import React, { useEffect, useState } from "react";
import DynamicChart from "@/components/dashboard/charts/DynamicChart";
import { fetchData } from "@/lib/api";
import Loader from "@/components/Loader";
type Props = {
  projectId: string;
};

const GcpCsqlCostByTime: React.FC<Props> = ({ projectId }) => {
  const [response, setResponse] = useState<any[]>([]);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const cloudPlatform = "gcp";
  const queryType = "gcp_csql_cost_by_time";

  useEffect(() => {
    const getData = async () => {
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

  if (loading)
    return (
      <div>
        <Loader />
      </div>
    );
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div className="h-full w-full bg-white rounded-lg shadow-md p-4 border flex flex-col">
      <h2 className="mb-4 text-lg">Csql Cost By Time</h2>
      <div className="flex-grow">
        <DynamicChart
          data={response}
          type="line"
          xKey="gcp_fact_dim.billing_period_start.day"
          yKey="gcp_fact_dim.total_list_cost"
          name="Csql Cost By Time"
          xAxisLabelFormatter={(label: string) =>
            label.length > 6 ? `${label.slice(0, 6)}...` : label
          }
          xAxisTooltipFormatter={(label: string) => label}
        />
      </div>
    </div>
  );
};

export default GcpCsqlCostByTime;
