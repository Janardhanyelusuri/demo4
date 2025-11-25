"use client";
import React, { useEffect, useState, useCallback, useRef } from "react";
import PersonaPieChartComponent from "./components/PersonaPieChartComponent";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import PersonaBarChartComponent from "./components/PersonaBarChartComponent";
import StackedBarChart from "./components/StackedBarChart";
import PersonaBudgetInfo from "./components/PersonaBudgetInfo";
import DatesInfo from "./components/DatesInfo";
import DashboardNameCard from "./components/DashboardNameCard";
import TagDateFilterBar from "@/components/filters/TagDateFilterBar";
import axiosInstance, { BACKEND } from "@/lib/api";
import { cn } from "@/lib/utils";
import Loader from "@/components/Loader";

interface Tags {
  tag_id: number;
  key: string;
  value: string;
  budget: number;
}

interface ProductOwnerDashboardProps {
  id: string;
  dashboardName: string;
  tag_id?: number; // Optional tag_id
}

const chartConfig = {
  pieCharts: [
    {
      title: "COST BY SERVICE NAMES",
      queryType: "consolidated_service_name_cost",
      nameField: "view_fact_billing.servicename",
      valueField: "view_fact_billing.total_billed_cost",
    },
    {
      title: "COST BY SERVICE CATEGORIES",
      queryType: "consolidated_cost_by_service_category",
      nameField: "view_dim_service.servicecategory",
      valueField: "view_fact_billing.total_billed_cost",
    },
  ],
  barCharts: [{
    title: "COST BY REGIONS",
    queryType: "consolidated_cost_by_region",
    nameField: "view_dim_region.regionname",
    valueField: "view_fact_billing.total_billed_cost",
  },
  {
    type: "bar",
    title: "VS COST BY INSTANCES",
    queryType: "azure_cost_by_instance_vs",
    nameField: "view_dim_resource.resourcename",
    valueField: "view_fact_billing.total_billed_cost",
  },],
  stackedBarChart: {
    title: "SERVICES COSTS BY DURATION",
    queryType: "consolidated_all_services_cost",
    
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

const ProductOwnerDashboard: React.FC<ProductOwnerDashboardProps> = ({
  id,
  dashboardName,
  tag_id
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
            <DashboardNameCard title="PRODUCT OWNER" />
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

          {/* Pie Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {chartConfig.pieCharts.map((chart, index) => (
              <div key={index} className="h-[400px] flex items-center justify-center">
                <div className="bg-white border border-[#E0E5EF] rounded-md shadow-md w-full h-full flex items-center justify-center p-4">
                  <PersonaPieChartComponent
                    title={chart.title}
                    queryType={chart.queryType}
                    id={id}
                    nameField={chart.nameField}
                    valueField={chart.valueField}
                    tag_id={selectedTagId}
                    duration={selectedDuration}
                    setParentLoading={setChildLoading}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Bar Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
            {/* First Bar Chart */}
            <div className="h-[500px] flex items-center justify-center">
              <div className="bg-white border border-[#E0E5EF] rounded-md shadow-md w-full h-full flex items-center justify-center p-4">
                <PersonaBarChartComponent
                  title={chartConfig.barCharts[0].title}
                  queryType={chartConfig.barCharts[0].queryType}
                  id={id}
                  nameField={chartConfig.barCharts[0].nameField}
                  valueField={chartConfig.barCharts[0].valueField}
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
                  title={chartConfig.barCharts[1].title}
                  queryType={chartConfig.barCharts[1].queryType}
                  id={id}
                  nameField={chartConfig.barCharts[1].nameField}
                  valueField={chartConfig.barCharts[1].valueField}
                  tag_id={selectedTagId}
                  duration={selectedDuration}
                  setParentLoading={setChildLoading}
                />
              </div>
            </div>
          </div>

          {/* Stacked Bar Chart */}
          <div className="h-[500px] flex items-center justify-center mt-4">
            <div className="bg-white border border-[#E0E5EF] rounded-md shadow-md w-full h-full flex items-center justify-center p-4">
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

export default ProductOwnerDashboard;

