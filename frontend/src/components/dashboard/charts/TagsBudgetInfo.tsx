"use client";
import React, { useEffect, useState } from "react";
import { fetchdDataDashboard, fetchDataTags, fetchResourceNames, fetchResourceIds } from "@/lib/api";
import {
  BarChartIcon,
  LineChartIcon,
  DollarSignIcon,
  TrendingUpIcon,
  TrendingDownIcon,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import apiQueue from "@/lib/apiQueueManager";
import Loader from "@/components/Loader";

interface ResourceNameResult {
  id?: string;
  name?: string;
  resourceId?: string;
  displayName?: string;
}

type TagsBudgetInfoProps = {
  cloudPlatform: string;
  queryType: string;
  projectId: string;
  title: string;
  tag_id?: number; // Added tag_id as an optional prop
  duration?: string;
  setParentLoading?: (isLoading: boolean) => void;
};

const TagsBudgetInfo: React.FC<TagsBudgetInfoProps> = ({
  cloudPlatform,
  queryType,
  projectId,
  title,
  tag_id, // Added tag_id to the destructured props
  duration,
  setParentLoading,
}) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (setParentLoading) setParentLoading(true);
    const getData = async () => {
      let retries = 0;
      const maxRetries = 3;
      while (retries < maxRetries) {
        try {
          if (cloudPlatform && queryType && projectId) {
            let resourceNameParam: string | undefined;
            if (tag_id) {
              let resourceResults;
              if (cloudPlatform === 'aws') {
                resourceResults = await apiQueue.enqueue(() => fetchResourceIds(tag_id), "ResourceIds");
              } else {
                resourceResults = await apiQueue.enqueue(() => fetchResourceNames(tag_id), "ResourceNames");
              }
              resourceNameParam = resourceResults?.[0]?.name || resourceResults?.[0]?.id;
            }
            const result = await apiQueue.enqueue(
              () => fetchDataTags(
                cloudPlatform,
                queryType,
                projectId.toString(),
                undefined,
                tag_id,
                duration,
                resourceNameParam,
              ),"FetchData");
            setData(result);
            setLoading(false);
            if (setParentLoading) setParentLoading(false);
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
            if (setParentLoading) setParentLoading(false);
          }
        }
      }
    };
    getData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cloudPlatform, queryType, projectId, tag_id, duration]);

  if (loading) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-[#fff] rounded-md shadow-md border border-[#E0E5EF]">
        <Loader />
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-[#fff] rounded-md shadow-md border border-[#E0E5EF] text-[#D82026]">
        Error: {error.message}
      </div>
    );
  }

  if (!data) {
    return null;
  }
  // Added the formatCurrency function
  const formatCurrency = (value: number) => {
    const absValue = Math.abs(value);
    const formatted = absValue.toFixed(1).toLocaleString();
    return value < 0 ? `-$${formatted}` : `$${formatted}`;
  };

  return (
    <div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Monthly Budget Card */}
        <Card className="relative w-full border border-[#E0E5EF] bg-[#fff] text-[#233E7D] overflow-hidden transition-all duration-300 hover:shadow-lg hover:scale-105">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-lg font-bold uppercase tracking-wide text-[#233E7D]">
                Monthly Cost: ${(data?.monthly_budget_actual_utilization ?? 0).toFixed(1).toLocaleString()}
              </div>
              <div className="w-8 h-8 rounded-full bg-[#E0E5EF] flex items-center justify-center text-[#233E7D]">
                <DollarSignIcon className="w-4 h-4" />
              </div>
            </div>
            <div className="flex items-center text-sm mt-4">
              <div className="text-gray-600">
              {data.monthly_budget_actual_utilization?.toFixed(1) ?? '0.0'}%
              </div>
              <span className="ml-1 text-[#6B7280]">Utilization</span>
            </div>
            <div className="text-sm font-bold mt-1 text-[#233E7D]">
              Monthly Budget: ${(data.monthly_budget ?? 0).toFixed(1).toLocaleString()}
            </div>
            <div className="flex items-center text-sm mt-4">
              <div className={`mr-1 ${
                data.monthly_budget_drift >= 0 && data.monthly_budget_drift <= 100
                  ? "text-[#22C55E]"
                  : "text-[#D82026]"
                }`}>
                {(data.monthly_budget_drift ?? 0).toFixed(1)}%
              </div>
              <span className="text-[#6B7280]">Drift</span>
            </div>
            <div className="text-sm font-bold mt-1 text-[#233E7D]">
              Actual Drift: {formatCurrency(data.monthly_budget_actual_drift ?? 0)}
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#233E7D] to-[#22C55E]"></div>
        </Card>

        {/* Quarterly Budget Card */}
        <Card className="relative w-full border border-[#E0E5EF] bg-[#fff] text-[#233E7D] overflow-hidden transition-all duration-300 hover:shadow-lg hover:scale-105">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-lg font-bold uppercase tracking-wide text-[#233E7D]">
                Quarterly Cost: ${(data.quarterly_budget_actual_utilization ?? 0).toFixed(1).toLocaleString()}
              </div>
              <div className="w-8 h-8 rounded-full bg-[#E0E5EF] flex items-center justify-center text-[#233E7D]">
                <DollarSignIcon className="w-4 h-4" />
              </div>
            </div>
            <div className="flex items-center text-sm mt-4">
              <div className="text-gray-600">
                {(data.quarterly_budget_actual_utilization ?? 0).toFixed(1)}%
              </div>
              <span className="ml-1 text-[#6B7280]">Utilization</span>
            </div>
            <div className="text-sm font-bold mt-1 text-[#233E7D]">
              Quarterly Budget: ${(data.quarterly_budget ?? 0).toFixed(1).toLocaleString()}
            </div>
            <div className="flex items-center text-sm mt-4">
              <div className={`mr-1 ${
                data.quarterly_budget_drift >= 0 && data.quarterly_budget_drift <= 100
                  ? "text-[#22C55E]"
                  : "text-[#D82026]"
                }`}>
                {(data.quarterly_budget_drift ?? 0).toFixed(1)}%
              </div>
              <span className="text-[#6B7280]">Drift</span>
            </div>
            <div className="text-sm font-bold mt-1 text-[#233E7D]">
              Actual Drift: {formatCurrency(data.quarterly_budget_actual_drift ?? 0)}
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#233E7D] to-[#22C55E]"></div>
        </Card>

        {/* Yearly Budget Card */}
        <Card className="relative w-full border border-[#E0E5EF] bg-[#fff] text-[#233E7D] overflow-hidden transition-all duration-300 hover:shadow-lg hover:scale-105">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-lg font-bold uppercase tracking-wide text-[#233E7D]">
                Yearly Cost: ${(data.yearly_budget_actual_utilization ?? 0).toFixed(1).toLocaleString()}
              </div>
              <div className="w-8 h-8 rounded-full bg-[#E0E5EF] flex items-center justify-center text-[#233E7D]">
                <DollarSignIcon className="w-4 h-4" />
              </div>
            </div>
            <div className="flex items-center text-sm mt-4">
              <div className="text-gray-600">
                {(data.yearly_budget_actual_utilization ?? 0).toFixed(1)}%
              </div>
              <span className="ml-1 text-[#6B7280]">Utilization</span>
            </div>
            <div className="text-sm font-bold mt-1 text-[#233E7D]">
              Yearly Budget: ${(data.yearly_budget ?? 0).toFixed(1).toLocaleString()}
            </div>
            <div className="flex items-center text-sm mt-4">
              <div className={`mr-1 ${
                data.yearly_budget_drift >= 0 && data.yearly_budget_drift <= 100
                  ? "text-[#22C55E]"
                  : "text-[#D82026]"
                }`}>
                {(data.yearly_budget_drift ?? 0).toFixed(1)}%
              </div>
              <span className="text-[#6B7280]">Drift</span>
            </div>
            <div className="text-sm font-bold mt-1 text-[#233E7D]">
              Actual Drift: {formatCurrency(data.yearly_budget_actual_drift ?? 0)}
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#233E7D] to-[#22C55E]"></div>
        </Card>

        {/* Forecast Card */}
        <Card className="relative w-full border border-[#E0E5EF] bg-[#fff] text-[#233E7D] overflow-hidden transition-all duration-300 hover:shadow-lg hover:scale-105">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-lg font-bold uppercase tracking-wide text-[#233E7D]">
                Forecast
              </div>
              <div className="w-8 h-8 rounded-full bg-[#E0E5EF] flex items-center justify-center text-[#233E7D]">
                <LineChartIcon className="w-4 h-4" />
              </div>
            </div>
            <div className="text-l font-bold mt-1 text-[#233E7D]">
              Monthly: ${(data.forecast_next_month_cost ?? 0).toFixed(1).toLocaleString()}
            </div>
            <div className="text-l font-bold mt-1 text-[#233E7D]">
              Quarterly: ${(data.forecast_next_quarter_cost ?? 0).toFixed(1).toLocaleString()}
            </div>
            <div className="text-l font-bold mt-1 text-[#233E7D]">
              Yearly: ${(data.forecast_next_year_cost ?? 0).toFixed(1).toLocaleString()}
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#233E7D] to-[#22C55E]"></div>
        </Card>
      </div>
    </div>
  );
};

export default TagsBudgetInfo;
