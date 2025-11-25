"use client";
import React, { useEffect, useState } from "react";
import { fetchdDataDashboard } from "@/lib/api";
import {
  BarChartIcon,
  LineChartIcon,
  DollarSignIcon,
  TrendingUpIcon,
  TrendingDownIcon,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import Loader from "@/components/Loader";

type DatesInfoProps = {
  queryType: string;
  id: string;
  title: string;
  tag_id?: number;  // Ensure tag_id is optional
};

import apiQueue from "@/lib/apiQueueManager";

const DatesInfo: React.FC<DatesInfoProps> = ({
  queryType,
  id,
  title,
  tag_id,
}) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const getData = async () => {
      let retries = 0;
      const maxRetries = 3;
      while (retries < maxRetries) {
        try {
          if (queryType && id) {
            const result = await apiQueue.enqueue(
              () => fetchdDataDashboard(
                queryType,
                id.toString(),
                tag_id  // Pass tag_id here if available
              ),
              "DatesInfo"
            );
            setData(result);
            setLoading(false);
            break;
          } else {
            throw new Error("Missing required parameters");
          }
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
  }, [queryType, id, tag_id]);  // Add tag_id to the dependency array

  if (loading) {
    return (
      <div className="h-full w-full flex items-center justify-center">
        <Loader />
      </div>
    );
  }

  if (error) {
    return <div>Error: {error.message}</div>;
  }

  if (!data) {
    return null;
  }

  return (
    <div>
      <div className="grid grid-cols-1 md:grid-cols-1 gap-6">
        <Card className="relative w-full border border-[#E0E5EF] bg-[#FFFFFF] text-[#233E7D] overflow-hidden transition-all duration-300 rounded-md shadow hover:shadow-lg hover:border-[#233E7D]">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex-grow"></div>
              <div className="text-lg font-bold tracking-tight text-right text-[#233E7D]">
                Duration: <span className="font-semibold">{new Date(data.earliest_charge_period_date).toLocaleDateString()} - {new Date(data.latest_charge_period_date).toLocaleDateString()}</span>
              </div>
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
        </Card>
      </div>
    </div>
  );
};

export default DatesInfo;
