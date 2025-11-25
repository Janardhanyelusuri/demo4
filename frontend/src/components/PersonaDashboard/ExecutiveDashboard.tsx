"use client";
import React, { useEffect, useState, useCallback, useRef } from "react";
import PersonaBarChartComponent from "./components/PersonaBarChartComponent";
import PersonaLineChartComponent from "./components/PersonaLineChartComponent";
import PersonaPieChartComponent from "./components/PersonaPieChartComponent";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { useRouter } from "next/navigation";
import PersonaBudgetInfo from "./components/PersonaBudgetInfo";
import DatesInfo from "./components/DatesInfo";
import DashboardNameCard from "./components/DashboardNameCard";
import TagDateFilterBar from "@/components/filters/TagDateFilterBar";
import axiosInstance, { BACKEND } from "@/lib/api";
import { cn } from "@/lib/utils";
import Loader from "@/components/Loader";

// ExecutiveDashboard-specific Tag interface
interface Tags {
  tag_id: number;
  key: string;
  value: string;
  budget: number;
}

interface ExecutiveDashboardProps {
  id: string;
  dashboardName: string;
  tag_id?: number; // Added tag_id as an optional prop
}

const chartConfig = {
  lineChart: {
    title: "COST OVER TIME",
    queryType: "consolidated_cost_trends_over_time",
    xKey: "view_dim_time.chargeperiodstart",
    yKey: "view_fact_billing.total_billed_cost",
    xAxisLabelFormatter: (label: string) =>
      label.length > 6 ? `${label.slice(0, 10)}` : label,
    xAxisTooltipFormatter: (label: string) => label,
  },
  pieChart: {
    title: "COST BY SERVICE NAME",
    queryType: "consolidated_service_name_cost",
    nameField: "view_fact_billing.servicename",
    valueField: "view_fact_billing.total_billed_cost",
    
  },
  barChart2: {
    title: "COST BY REGIONS",
    queryType: "consolidated_cost_by_region",
    nameField: "view_dim_region.regionname",
    valueField: "view_fact_billing.total_billed_cost",
  },
};

const cardClass =
  "bg-white rounded-2xl shadow-lg border border-brand-blue/10 p-6 mb-4";
const headingClass =
  "text-brand-blue text-2xl md:text-3xl font-extrabold tracking-tight mb-2";
const subheadingClass =
  "text-brand-dark text-lg font-semibold mb-1";
const labelClass = "text-brand-gray font-medium";
const valueClass = "text-brand-blue font-bold text-xl";
const buttonClass =
  "bg-brand-blue hover:bg-brand-blue-dark text-white font-semibold rounded-lg px-4 py-2 transition-colors duration-200 shadow-md";

const ExecutiveDashboard: React.FC<ExecutiveDashboardProps> = ({
  id,
  dashboardName,
  tag_id
}) => {
  const router = useRouter();
  const [selectedTagId, setSelectedTagId] = useState<number | undefined>(undefined);
  const [selectedDuration, setSelectedDuration] = useState<string | undefined>(undefined);
  
  // Global loading state
  const [globalLoading, setGlobalLoading] = useState(true);

  // Track how many children are loading
  const [loadingCount, setLoadingCount] = useState(0);
  const setChildLoading = useCallback((isLoading: boolean) => {
    setLoadingCount((prev) => {
      const next = isLoading ? prev + 1 : Math.max(prev - 1, 0);
      return next;
    });
  }, []);

  // Show loader while any child is loading
  useEffect(() => {
    setGlobalLoading(loadingCount > 0);
  }, [loadingCount]);

  // Set tag ID from prop if present
  const hasSetTagId = useRef(false);
  useEffect(() => {
    if (hasSetTagId.current) return;
    hasSetTagId.current = true;
    if (tag_id) {
      setSelectedTagId(tag_id);
    }
  }, [tag_id]);

  const handleBackClick = () => {
    router.push("/dashboard-home");
  };

  // Handles both tag and date filter changes
  const handleTagSelect = (tagId: number | undefined) => {
    setSelectedTagId(tagId);
  };

  const handleDateRangeChange = (range: string | undefined) => {
    console.log("Date range selected:", range);
    setSelectedDuration(range);
  };

  // Debug effect to track selectedDuration changes
  useEffect(() => {
    console.log("selectedDuration changed to:", selectedDuration);
  }, [selectedDuration]);

  function handleTagChange(tagId: number | undefined): void {
    setSelectedTagId(tagId);
  }
  return (
    <div className="bg-[#F9FEFF] min-h-screen h-screen overflow-auto p-6 font-sans">
      <TagDateFilterBar
        onTagSelect={handleTagSelect}
        onDateRangeSelect={handleDateRangeChange}
      />
      <div className="relative">
        {globalLoading && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-white bg-opacity-60">
            <Loader />
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <div className="flex-grow">
            <DashboardNameCard title="EXECUTIVE" />
          </div>
          <div className="mb-4 flex-shrink-0">
            <DatesInfo
              queryType="consolidated_charge_period_dates"
              id={id}
              title="Duration"
            />
          </div>
        </div>

        <div className="mb-4">
          <PersonaBudgetInfo
            queryType="consolidated_budgets_data"
            id={id}
            title="Budget"
            tag_id={selectedTagId}
          />
        </div>

        {/* Pie Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="h-[400px] flex items-center justify-center">
            <div className="bg-white border border-[#E0E5EF] rounded-md shadow-md w-full h-full flex items-center justify-center p-4">
              <PersonaPieChartComponent
                title={chartConfig.pieChart.title}
                queryType={chartConfig.pieChart.queryType}
                id={id}
                nameField={chartConfig.pieChart.nameField}
                valueField={chartConfig.pieChart.valueField}
                tag_id={selectedTagId}
                duration={selectedDuration}
                setParentLoading={setChildLoading}
              />
            </div>
          </div>
        </div>

        {/* Bar Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
          {/* First Bar Chart */}
          <div className="h-[500px] flex items-center justify-center">
            <div className="bg-white border border-[#E0E5EF] rounded-md shadow-md w-full h-full flex items-center justify-center p-4">
              <PersonaBarChartComponent
                title={chartConfig.barChart2.title}
                queryType={chartConfig.barChart2.queryType}
                id={id}
                nameField={chartConfig.barChart2.nameField}
                valueField={chartConfig.barChart2.valueField}
                tag_id={selectedTagId}
                duration={selectedDuration}
                setParentLoading={setChildLoading}
              />
            </div>
          </div>
          {/* Second Bar Chart */}
          <div className="h-[500px] flex items-center justify-center">
            <div className="bg-white border border-[#E0E5EF] rounded-md shadow-md w-full h-full flex items-center justify-center p-4">
              <PersonaBarChartComponent
                title={chartConfig.barChart2.title}
                queryType={chartConfig.barChart2.queryType}
                id={id}
                nameField={chartConfig.barChart2.nameField}
                valueField={chartConfig.barChart2.valueField}
                tag_id={selectedTagId}
                duration={selectedDuration}
                setParentLoading={setChildLoading}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExecutiveDashboard;