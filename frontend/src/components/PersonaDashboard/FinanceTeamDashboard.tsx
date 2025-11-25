import React, { useEffect, useState, useCallback, useRef } from "react";
import PersonaLineChartComponent from "./components/PersonaLineChartComponent";
import PersonaPieChartComponent from "./components/PersonaPieChartComponent";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import PersonaBarChartComponent from "./components/PersonaBarChartComponent";
import PersonaBudgetInfo from "./components/PersonaBudgetInfo";
import DatesInfo from "./components/DatesInfo";
import DashboardNameCard from "./components/DashboardNameCard";
import TagDateFilterBar from "@/components/filters/TagDateFilterBar";
import axiosInstance, { BACKEND } from "@/lib/api";
import Loader from "@/components/Loader";

// Tag interface
interface Tags {
  tag_id: number;
  key: string;
  value: string;
  budget: number;
}

interface FinanceTeamDashboardProps {
  id: string;
  dashboardName: string;
  tag_id?: number; // Add tag_id as optional prop
}

// Unified chart configuration
const chartConfig = {
  lineChart: {
    title: "COST OVER TIME",
    queryType: "consolidated_cost_trends_over_time",
    xKey: "view_dim_time.chargeperiodstart",
    yKey: "view_fact_billing.total_billed_cost",
  },
  pieChart: {
    title: "COST BY SERVICE NAMES",
    queryType: "consolidated_service_name_cost",
    nameField: "view_fact_billing.servicename",
    valueField: "view_fact_billing.total_billed_cost",
  },
  barChart: {
    title: "COST BY SERVICE CATEGORIES",
    queryType: "consolidated_cost_by_service_category",
    nameField: "view_dim_service.servicecategory",
    valueField: "view_fact_billing.total_billed_cost",
  },
  barChart2: {
    title: "COST BY REGIONS",
    queryType: "consolidated_cost_by_region",
    nameField: "view_dim_region.regionname",
    valueField: "view_fact_billing.total_billed_cost",
  },
};

const FinanceTeamDashboard: React.FC<FinanceTeamDashboardProps> = ({
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
      <header className="flex justify-between items-center mb-6 sticky top-0 z-30 bg-[#F9FEFF]">
        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={handleBackClick} className="p-2 rounded-full hover:bg-[#E0E5EF]">
            <ArrowLeft size={24} className="text-[#233E7D]" />
          </Button>
          <h1 className="text-3xl font-bold text-[#233E7D] tracking-tight">{dashboardName}: Finance Team</h1>
        </div>
      </header>

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
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div className="flex-grow">
              <DashboardNameCard title="FINANCE TEAM" />
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
              setParentLoading={setChildLoading}
            />
          </div>

          {/* Pie and Bar Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            <div className="h-[400px] items-center justify-center flex">
              <PersonaPieChartComponent
                title={chartConfig.pieChart.title}
                queryType={chartConfig.pieChart.queryType}
                id={id}
                nameField={chartConfig.pieChart.nameField}
                valueField={chartConfig.pieChart.valueField}
                tag_id={selectedTagId}
                duration={selectedDuration}
              />
            </div>
            <div className="h-[400px] items-center justify-center flex">
              <PersonaBarChartComponent
                title={chartConfig.barChart.title}
                queryType={chartConfig.barChart.queryType}
                id={id}
                nameField={chartConfig.barChart.nameField}
                valueField={chartConfig.barChart.valueField}
                tag_id={selectedTagId}
                duration={selectedDuration}
              />
            </div>
          </div>

                              <div className="h-[400px] items-center justify-center flex">
            <PersonaLineChartComponent
              title={chartConfig.lineChart.title}
              queryType={chartConfig.lineChart.queryType}
              id={id}
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
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="h-[400px] items-center justify-center flex">
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

            <div className="h-[400px] items-center justify-center flex">
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
  );
};

export default FinanceTeamDashboard;