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

type BudgetInfoProps = {
  queryType: string;
  id: string;
  title: string;
  tag_id?: number;
};
import apiQueue from "@/lib/apiQueueManager"
const BudgetInfo: React.FC<BudgetInfoProps> = ({
  queryType,
  id,
  title,
}) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const getData = async () => {
      const maxRetries = 3;
      let retries = 0;
      let success = false;
      while (retries < maxRetries && !success) {
        try {
          if (queryType && id) {
            const result = await apiQueue.enqueue(
              () => fetchdDataDashboard(
              queryType,
              id.toString()
            ), "BudgetInfoCard");
            setData(result);
            setLoading(false);
            success = true;
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
  }, [queryType, id]);


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
    // Added the formatCurrency fuction
const formatCurrency = (value: number) => {
  const absValue = Math.abs(value);
  const formatted = absValue.toFixed(1).toLocaleString();
  return value < 0 ? `-$${formatted}` : `$${formatted}`;
};

  return (
    <div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Monthly Budget Card */}
        <Card className="relative w-full border border-[#E0E5EF] bg-[#FFFFFF] text-[#233E7D] overflow-hidden transition-all duration-300 rounded-md shadow hover:shadow-lg hover:border-[#233E7D]">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-lg font-bold tracking-tight text-[#233E7D]">
                Monthly Cost: ${data.monthly_budget_actual_utilization.toFixed(1).toLocaleString()}
              </div>
              <div className="w-8 h-8 rounded-full bg-[#F9FEFF] flex items-center justify-center text-[#233E7D] border border-[#E0E5EF]">
                <DollarSignIcon className="w-4 h-4" />
              </div>
            </div>
            <div className="flex items-center text-sm mt-4">
              <div className="text-[#6B7280] font-medium">
                {data.monthly_budget_utilization.toFixed(1)}%
              </div>
              <span className="ml-1 text-[#6B7280]">Utilization</span>
            </div>
            <div className="text-sm font-bold mt-1 text-[#233E7D]">
              Monthly Budget: ${data.monthly_budget.toFixed(1).toLocaleString()}
            </div>
            <div className="flex items-center text-sm mt-4">
              <div className={`mr-1 font-semibold ${
                data.monthly_budget_drift >= 0 && data.monthly_budget_drift <= 100
                  ? "text-[#22C55E]"
                  : "text-[#D82026]"
                }`}>
                {data.monthly_budget_drift.toFixed(1)}%
              </div>
              <span className="text-[#6B7280]">Drift</span>
            </div>
            <div className="text-sm font-bold mt-1 text-[#233E7D]">
              Actual Drift: {formatCurrency(data.quarterly_budget_actual_drift ?? 0)}
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
        </Card>

        {/* Quarterly Budget Card */}
        <Card className="relative w-full border border-[#E0E5EF] bg-[#FFFFFF] text-[#233E7D] overflow-hidden transition-all duration-300 rounded-md shadow hover:shadow-lg hover:border-[#233E7D]">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-lg font-bold tracking-tight text-[#233E7D]">
                Quarterly Cost: ${data.quarterly_budget_actual_utilization.toFixed(1).toLocaleString()}
              </div>
              <div className="w-8 h-8 rounded-full bg-[#F9FEFF] flex items-center justify-center text-[#233E7D] border border-[#E0E5EF]">
                <DollarSignIcon className="w-4 h-4" />
              </div>
            </div>
            <div className="flex items-center text-sm mt-4">
              <div className="text-[#6B7280] font-medium">
                {data.quarterly_budget_utilization.toFixed(1)}%
              </div>
              <span className="ml-1 text-[#6B7280]">Utilization</span>
            </div>
            <div className="text-sm font-bold mt-1 text-[#233E7D]">
              Quarterly Budget: ${data.quarterly_budget.toFixed(1).toLocaleString()}
            </div>
            <div className="flex items-center text-sm mt-4">
              <div className={`mr-1 font-semibold ${
                data.quarterly_budget_drift >= 0 && data.quarterly_budget_drift <= 100
                  ? "text-[#22C55E]"
                  : "text-[#D82026]"
                }`}>
                {data.quarterly_budget_drift.toFixed(1)}%
              </div>
              <span className="text-[#6B7280]">Drift</span>
            </div>
            <div className="text-sm font-bold mt-1 text-[#233E7D]">
              Actual Drift: {formatCurrency(data.quarterly_budget_actual_drift ?? 0)}
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
        </Card>

        {/* Yearly Budget Card */}
        <Card className="relative w-full border border-[#E0E5EF] bg-[#FFFFFF] text-[#233E7D] overflow-hidden transition-all duration-300 rounded-md shadow hover:shadow-lg hover:border-[#233E7D]">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-lg font-bold tracking-tight text-[#233E7D]">
                Yearly Cost: ${data.yearly_budget_actual_utilization.toFixed(1).toLocaleString()}
              </div>
              <div className="w-8 h-8 rounded-full bg-[#F9FEFF] flex items-center justify-center text-[#233E7D] border border-[#E0E5EF]">
                <DollarSignIcon className="w-4 h-4" />
              </div>
            </div>
            <div className="flex items-center text-sm mt-4">
              <div className="text-[#6B7280] font-medium">
                {data.yearly_budget_utilization.toFixed(1)}%
              </div>
              <span className="ml-1 text-[#6B7280]">Utilization</span>
            </div>
            <div className="text-sm font-bold mt-1 text-[#233E7D]">
              Yearly Budget: ${data.yearly_budget.toFixed(1).toLocaleString()}
            </div>
            <div className="flex items-center text-sm mt-4">
              <div className={`mr-1 font-semibold ${
                data.yearly_budget_drift >= 0 && data.yearly_budget_drift <= 100
                  ? "text-[#22C55E]"
                  : "text-[#D82026]"
                }`}>
                {data.yearly_budget_drift.toFixed(1)}%
              </div>
              <span className="text-[#6B7280]">Drift</span>
            </div>
            <div className="text-sm font-bold mt-1 text-[#233E7D]">
              Actual Drift: {formatCurrency(data.quarterly_budget_actual_drift ?? 0)}
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
        </Card>

        {/* Forecast Card */}
        <Card className="relative w-full border border-[#E0E5EF] bg-[#FFFFFF] text-[#233E7D] overflow-hidden transition-all duration-300 rounded-md shadow hover:shadow-lg hover:border-[#233E7D]">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-lg font-bold tracking-tight text-[#233E7D]">
                Forecast
              </div>
              <div className="w-8 h-8 rounded-full bg-[#F9FEFF] flex items-center justify-center text-[#233E7D] border border-[#E0E5EF]">
                <LineChartIcon className="w-4 h-4" />
              </div>
            </div>
            <div className="text-base font-bold mt-1 text-[#233E7D]">
              Monthly: ${data.forecast_next_month_cost.toFixed(1).toLocaleString()}
            </div>
            <div className="text-base font-bold mt-1 text-[#233E7D]">
              Quarterly: ${data.forecast_next_quarter_cost.toFixed(1).toLocaleString()}
            </div>
            <div className="text-base font-bold mt-1 text-[#233E7D]">
              Yearly: ${data.forecast_next_year_cost.toFixed(1).toLocaleString()}
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
        </Card>
      </div>
    </div>
  );
};

export default BudgetInfo;
