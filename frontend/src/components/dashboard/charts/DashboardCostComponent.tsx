import React, { useState, useEffect } from "react";
import { fetchDashboardData, fetchData } from "@/lib/api";
import CardComponent from "../CardComponent";
import Loader from "@/components/Loader";
import { CircleAlert } from "lucide-react";
import apiQueue from "@/lib/apiQueueManager";

interface DashboardCostComponentProps {
  cloudPlatform: string;
  queryType: string;
  costField: string;
  label: string;
  dashboardId: string;
}

const DashboardCostComponent: React.FC<DashboardCostComponentProps> = ({
  cloudPlatform,
  queryType,
  costField,
  label,
  dashboardId,
}) => {
  const [response, setResponse] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const getData = async (retries = 3) => {
      if (!cloudPlatform) return;

      try {
        const result = await apiQueue.enqueue(
          () => fetchDashboardData(
          cloudPlatform,
          queryType,
          dashboardId
        ),"DCostComponent");
        setResponse(result.data);
        setLoading(false);
      } catch (error) {
        if (retries > 0) {
          console.log(`Retrying... ${retries} attempts left`);
          await getData(retries - 1);
        } else {
          setError(
            error instanceof Error
              ? error
              : new Error("An unknown error occurred")
          );
          setLoading(false);
        }
      }
    };

    getData();
  }, [cloudPlatform, queryType, dashboardId]);

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

  const totalCostUSD = response
    .reduce((sum: number, item: any) => {
      const cost = parseFloat(item[costField]) || 0;
      return sum + cost;
    }, 0)
    .toFixed(1);

  return <CardComponent label={label} value={totalCostUSD} />;
};

export default DashboardCostComponent;
