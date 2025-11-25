"use client";
import React, { useEffect, useState } from "react";
import { fetchdDataDashboard } from "@/lib/api";
import {
  LineChartIcon,
  DollarSignIcon,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import Loader from "@/components/Loader";

type PersonaBudgetInfoProps = {
  queryType: string;
  id: string;
  title: string;
  tag_id?: number;
  setParentLoading?: (isLoading: boolean) => void;
};
import apiQueue from "@/lib/apiQueueManager"
const PersonaBudgetInfo: React.FC<PersonaBudgetInfoProps> = ({
  queryType,
  id,
  title,
  tag_id,
  setParentLoading
}) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (setParentLoading) setParentLoading(true);
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
              id.toString(),
              tag_id
            ), "PersonaBudgetInfoCard");
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
      if (setParentLoading) setParentLoading(false);
    };
    getData();
  }, [queryType, id, tag_id, setParentLoading]);

  const safeFormatNumber = (value: any, decimals: number = 2): string => {
    if (value === null || value === undefined || isNaN(Number(value))) {
      return '0.00';
    }
    return Number(value).toFixed(decimals);
  };

  // Format currency with sign handling
  const formatCurrency = (value: number) => {
    if (value === null || value === undefined) return '$0.0';
    const absValue = Math.abs(value);
    const formatted = absValue.toLocaleString(undefined, {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1
    });
    return value < 0 ? `-$${formatted}` : `$${formatted}`;
  };

  if (loading) {
    return (
      <div className="h-full w-full flex items-center justify-center">
        <Loader />
      </div>);
  }

  if (error) {
    return <div>Error: {error.message}</div>;
  }

  if (!data) {
    return null;
  }

  return (
    <div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Monthly Budget Card */}
        <Card className="relative w-full border border-[#E0E5EF] bg-[#FFFFFF] text-[#233E7D] overflow-hidden transition-all duration-300 rounded-md shadow hover:shadow-lg hover:border-[#233E7D]">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-2xl font-bold tracking-tight text-[#233E7D]">
                Monthly Cost
              </div>
              <div className="w-8 h-8 rounded-full bg-[#F9FEFF] flex items-center justify-center text-[#233E7D] border border-[#E0E5EF]">
                <DollarSignIcon className="w-4 h-4" />
              </div>
            </div>
            <div className="text-2xl font-bold text-[#233E7D] mt-6">
              {formatCurrency(data.monthly_budget_actual_utilization)}
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
        </Card>

        {/* Quarterly Budget Card */}
        <Card className="relative w-full border border-[#E0E5EF] bg-[#FFFFFF] text-[#233E7D] overflow-hidden transition-all duration-300 rounded-md shadow hover:shadow-lg hover:border-[#233E7D]">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-2xl font-bold tracking-tight text-[#233E7D]">
                Quarterly Cost
              </div>
              <div className="w-8 h-8 rounded-full bg-[#F9FEFF] flex items-center justify-center text-[#233E7D] border border-[#E0E5EF]">
                <DollarSignIcon className="w-4 h-4" />
              </div>
            </div>
            <div className="text-2xl font-bold text-[#233E7D] mt-6">
              {formatCurrency(data.quarterly_budget_actual_utilization)}
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
        </Card>

        {/* Yearly Budget Card */}
        <Card className="relative w-full border border-[#E0E5EF] bg-[#FFFFFF] text-[#233E7D] overflow-hidden transition-all duration-300 rounded-md shadow hover:shadow-lg hover:border-[#233E7D]">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-2xl font-bold tracking-tight text-[#233E7D]">
                Yearly Cost
              </div>
              <div className="w-8 h-8 rounded-full bg-[#F9FEFF] flex items-center justify-center text-[#233E7D] border border-[#E0E5EF]">
                <DollarSignIcon className="w-4 h-4" />
              </div>
            </div>
            <div className="text-2xl font-bold text-[#233E7D] mt-6">
              {formatCurrency(data.yearly_budget_actual_utilization)}
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
        </Card>

        {/* Forecast Card */}
        <Card className="relative w-full border border-[#E0E5EF] bg-[#FFFFFF] text-[#233E7D] overflow-hidden transition-all duration-300 rounded-md shadow hover:shadow-lg hover:border-[#233E7D]">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-2xl font-bold tracking-tight text-[#233E7D]">
                Forecast
              </div>
              <div className="w-8 h-8 rounded-full bg-[#F9FEFF] flex items-center justify-center text-[#233E7D] border border-[#E0E5EF]">
                <LineChartIcon className="w-4 h-4" />
              </div>
            </div>
            <div className="text-base font-bold text-[#233E7D] mt-6">
              Monthly: {formatCurrency(data.forecast_next_month_cost)}
            </div>
            <div className="text-base font-bold text-[#233E7D] mt-2">
              Quarterly: {formatCurrency(data.forecast_next_quarter_cost)}
            </div>
            <div className="text-base font-bold text-[#233E7D] mt-2">
              Yearly: {formatCurrency(data.forecast_next_year_cost)}
            </div>
          </CardContent>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-[#D82026] to-[#233E7D]" />
        </Card>
      </div>
    </div>
  );
};

export default PersonaBudgetInfo;
