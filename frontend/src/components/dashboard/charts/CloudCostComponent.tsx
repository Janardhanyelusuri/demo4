import React, { useState, useEffect } from "react";
import { fetchData } from "@/lib/api";
import CardComponent from "../CardComponent";
import Loader from "@/components/Loader";
import { CircleAlert } from "lucide-react";
import apiQueue from "@/lib/apiQueueManager";

interface CloudCostComponentProps {
  cloudPlatform: string;
  queryType: string;
  costField: string;
  label: string;
  projectId: string;
  service_names?: string;
  tag_id?: number;
  duration?: string; // Optional duration parameter
}

const CloudCostComponent: React.FC<CloudCostComponentProps> = ({
  cloudPlatform,
  queryType,
  costField,
  label,
  projectId,
  service_names,
  tag_id,
  duration,
}) => {
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const getData = async () => {
      if (!cloudPlatform) return;

      setLoading(true);
      setError(null);

      let retries = 0;
      const maxRetries = 3;

      while (retries < maxRetries) {
        try {
          const result = await apiQueue.enqueue(
            () =>
              fetchData(
                cloudPlatform,
                queryType,
                projectId,
                undefined,
                undefined,
                service_names,
                tag_id, // pass tag_id here
                duration
              ),
            "CloudCost"
          );
          setResponse(result);
          setLoading(false);
          return;
        } catch (error) {
          retries++;
          if (retries === maxRetries) {
            setError(
              error instanceof Error
                ? error
                : new Error("An unknown error occurred")
            );
            setLoading(false);
          }
        }
      }
    };

    getData();
  }, [cloudPlatform, queryType, projectId, service_names, tag_id, duration]); // include tag_id in dependencies

  if (loading)
    return (
      <div className="bg-[#fff] p-4 rounded-md shadow-md border border-[#E0E5EF] flex items-center justify-center min-h-[80px]">
        <Loader />
      </div>
    );

  if (error)
    return (
      <div className="bg-[#fff] p-4 rounded-md shadow-md border border-[#E0E5EF] flex items-center justify-center min-h-[80px] text-[#D82026]">
        <CircleAlert className="w-5 h-5 mr-2" />
        <span>Error loading cost</span>
      </div>
    );

  let totalCostUSD = "0.0";

  if (response && response.data && Array.isArray(response.data)) {
    const costData = response.data[0];
    if (costData && costField in costData) {
      totalCostUSD = parseFloat(costData[costField]).toFixed(1);
    }
  } else {
    console.error("Unexpected response format:", response);
  }

  return <CardComponent label={label} value={totalCostUSD} />;
};

export default CloudCostComponent;
