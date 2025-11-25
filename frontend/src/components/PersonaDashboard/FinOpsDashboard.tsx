import React, { useEffect, useState, useCallback, useRef } from "react";
import PersonaLineChartComponent from "./components/PersonaLineChartComponent";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { useRouter } from "next/navigation";
import PersonaBudgetInfo from "./components/PersonaBudgetInfo";
import PersonaPieChartComponent from "./components/PersonaPieChartComponent";
import PersonaBarChartComponent from "./components/PersonaBarChartComponent";
import DatesInfo from "./components/DatesInfo";
import DashboardNameCard from "./components/DashboardNameCard";
import TagDateFilterBar from "@/components/filters/TagDateFilterBar";
import axiosInstance, { BACKEND } from "@/lib/api";
import StackedBarChart from "./components/StackedBarChart";
import Loader from "@/components/Loader";

interface Tags {
  tag_id: number;
  key: string;
  value: string;
  budget: number;
}

interface FinOpsDashboardProps {
  id?: string;
  dashboardName?: string;
  tag_id?: number;
}

const chartConfig = {
  lineChart: {
    title: "COST OVER TIME",
    queryType: "consolidated_cost_trends_over_time",
    xKey: "view_dim_time.chargeperiodstart",
    yKey: "view_fact_billing.total_billed_cost",
  },
  pieChart: {
    title: "COST BY SERVICE CATEGORIES",
    queryType: "consolidated_cost_by_service_category",
    nameField: "view_dim_service.servicecategory",
    valueField: "view_fact_billing.total_billed_cost",
  },
  barChart: {
    title: "COST BY SERVICE NAMES",
    queryType: "consolidated_service_name_cost",
    nameField: "view_fact_billing.servicename",
    valueField: "view_fact_billing.total_billed_cost",
  },
  barChart2: {
    title: "COST BY SKU ID",
    queryType: "azure_sku_id_cost",
    nameField: "view_dim_service.skuid",
    valueField: "view_fact_billing.total_billed_cost",
  },
  stackedBarChart: {
    title: "SERVICES COSTS BY DURATION",
    queryType: "consolidated_all_services_cost",
  },
};

const FinOpsDashboard: React.FC<FinOpsDashboardProps> = ({
  id,
  dashboardName,
  tag_id,
}) => {
  const router = useRouter();
  const [selectedTagId, setSelectedTagId] = useState<number | undefined>(undefined);
  const [selectedDuration, setSelectedDuration] = useState<string | undefined>(undefined);
  
  // Global loading state similar to AtAGlance
  const [globalLoading, setGlobalLoading] = useState(false);
  const [loadingCount, setLoadingCount] = useState(0);
  const setChildLoading = useCallback((isLoading: boolean) => {
    setLoadingCount((prev) => {
      const next = isLoading ? prev + 1 : Math.max(prev - 1, 0);
      return next;
    });
  }, []);
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

  // Unified handler for tag and date filter changes
  const handleTagSelect = (tagId: number | undefined) => {
    setSelectedTagId(tagId);
  };

  const handleDateRangeChange = (range: string | undefined) => {
    console.log("Date range selected:", range);
    console.log("Setting selectedDuration to:", range);
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
    <div className="bg-[#F9FEFF] p-6 min-h-screen overflow-auto font-sans">
      <TagDateFilterBar
        onTagSelect={handleTagChange}
        onDateRangeSelect={handleDateRangeChange}
      />

      {/* Main dashboard content with loader overlay */}
      <div className="relative">
        {globalLoading && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-white bg-opacity-60">
            <Loader />
          </div>
        )}
        <div className="bg-white border border-[#E0E5EF] shadow-md rounded-md p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <DashboardNameCard title="FINOPS" />
            <DatesInfo
              queryType="consolidated_charge_period_dates"
              id={id as string}
              title="Duration"
            />
          </div>

          <PersonaBudgetInfo
            queryType="consolidated_budgets_data"
            id={id as string}
            title="Budget"
            tag_id={selectedTagId}
            setParentLoading={setChildLoading}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <PersonaLineChartComponent
              title={chartConfig.lineChart.title}
              queryType={chartConfig.lineChart.queryType}
              id={id as string}
              xKey={chartConfig.lineChart.xKey}
              yKey={chartConfig.lineChart.yKey}
              name={chartConfig.lineChart.title}
              xAxisLabelFormatter={(label: string) =>
                label.length > 6 ? `${label.slice(0, 6)}...` : label
              }
              xAxisTooltipFormatter={(label: string) => label}
              tag_id={selectedTagId}
              duration={selectedDuration}
              setParentLoading={setChildLoading}
            />
            <PersonaBarChartComponent
              title={chartConfig.barChart2.title}
              queryType={chartConfig.barChart2.queryType}
              id={id as string}
              nameField={chartConfig.barChart2.nameField}
              valueField={chartConfig.barChart2.valueField}
              tag_id={selectedTagId}
              duration={selectedDuration}
              setParentLoading={setChildLoading}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="min-h-[450px] w-full flex justify-center items-center">
              <PersonaPieChartComponent
                title={chartConfig.pieChart.title}
                queryType={chartConfig.pieChart.queryType}
                id={id as string}
                nameField={chartConfig.pieChart.nameField}
                valueField={chartConfig.pieChart.valueField}
                tag_id={selectedTagId}
                duration={selectedDuration}
                setParentLoading={setChildLoading}
              />
            </div>
            <div className="min-h-[450px] w-full flex justify-center items-center">
              <PersonaBarChartComponent
                title={chartConfig.barChart.title}
                queryType={chartConfig.barChart.queryType}
                id={id as string}
                nameField={chartConfig.barChart.nameField}
                valueField={chartConfig.barChart.valueField}
                tag_id={selectedTagId}
                duration={selectedDuration}
                setParentLoading={setChildLoading}
              />
            </div>
          </div>

          <div className="h-[500px]">
            <StackedBarChart
              title={chartConfig.stackedBarChart.title}
              queryType={chartConfig.stackedBarChart.queryType}
              id={id}
              tag_id={selectedTagId}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default FinOpsDashboard;
