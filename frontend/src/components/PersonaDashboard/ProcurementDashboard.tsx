"use client";
import React, { useEffect, useState, useCallback } from "react";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import PersonaBarChartComponent from "./components/PersonaBarChartComponent";
import PersonaPieChartComponent from "./components/PersonaPieChartComponent";
import PersonaBudgetInfo from "./components/PersonaBudgetInfo";
import DatesInfo from "./components/DatesInfo";
import DashboardNameCard from "./components/DashboardNameCard";
import TagDateFilterBar from "@/components/filters/TagDateFilterBar";
import axiosInstance, { BACKEND } from "@/lib/api";
import { cn } from "@/lib/utils";
import Loader from "@/components/Loader";

// Tag interface
interface Tags {
  tag_id: number;
  key: string;
  value: string;
  budget: number;
}

// Cloud Platform Configuration
const cloudPlatformConfig: Record<string, any> = {
  commonCharts: [
    {
      type: "bar",
      title: "COST BY REGIONS",
      queryType: "consolidated_cost_by_region",
      nameField: "view_dim_region.regionname",
      valueField: "view_fact_billing.total_billed_cost",
    },
    {
      type: "pie",
      title: "COST BY SERVICE NAMES",
      queryType: "consolidated_service_name_cost",
      nameField: "view_fact_billing.servicename",
      valueField: "view_fact_billing.total_billed_cost",
    },
  ],
  azure: {
    specificCharts: [
      {
        type: "pie",
        title: "COST BY SKU ID",
        queryType: "azure_sku_id_cost",
        nameField: "view_dim_service.skuid",
        valueField: "view_fact_billing.total_billed_cost",
      },
      {
        type: "bar",
        title: "VS COST BY INSTANCES",
        queryType: "azure_cost_by_instance_vs",
        nameField: "view_dim_resource.resourcename",
        valueField: "view_fact_billing.total_billed_cost",
      },
      {
        type: "bar",
        title: "CONTAINER REGISTRY COST BY INSTANCES",
        queryType: "azure_cost_by_instance_container_registry",
        nameField: "view_dim_resource.resourcename",
        valueField: "view_fact_billing.total_billed_cost",
      },
      {
        type: "bar",
        title: "ML COST BY INSTANCE",
        queryType: "azure_cost_by_instance_ml",
        nameField: "view_dim_resource.resourcename",
        valueField: "view_fact_billing.total_billed_cost",
      },
    ],
  },
  aws: {
    specificCharts: [
      {
        type: "bar",
        title: "EC2 COST BY INSTANCE",
        queryType: "aws_cost_by_instance_ecc",
        nameField: "view_fact_billing.servicename",
        valueField: "view_fact_billing.total_billed_cost",
      },
      {
        type: "bar",
        title: "LOAD BALANCER COST BY INSTANCES",
        queryType: "aws_cost_by_instance_load_balancing",
        nameField: "view_fact_billing.servicename",
        valueField: "view_fact_billing.total_billed_cost",
      },
      {
        type: "bar",
        title: "CLOUD WATCH COST BY INSTANCES",
        queryType: "aws_cost_by_instance_cloud_watch",
        nameField: "view_fact_billing.servicename",
        valueField: "view_fact_billing.total_billed_cost",
      },
    ],
  },
  gcp: {
    specificCharts: [
      {
        type: "bar",
        title: "CLOUD SQL COST BY INSTANCES",
        queryType: "gcp_cost_by_instance_cloud_sql",
        nameField: "view_fact_billing.servicename",
        valueField: "view_fact_billing.total_billed_cost",
      },
      {
        type: "bar",
        title: "KUBERNETES ENGINE COST BY INSTANCES",
        queryType: "gcp_cost_by_instance_kubernetes_engine",
        nameField: "view_fact_billing.servicename",
        valueField: "view_fact_billing.total_billed_cost",
      },
      {
        type: "bar",
        title: "COMPUTE ENGINE COST BY INSTANCES",
        queryType: "gcp_cost_by_instance_compute_engine",
        nameField: "view_fact_billing.servicename",
        valueField: "view_fact_billing.total_billed_cost",
      },
      {
        type: "bar",
        title: "NETWORKING COST BY INSTANCES",
        queryType: "gcp_cost_by_instance_networking",
        nameField: "view_fact_billing.servicename",
        valueField: "view_fact_billing.total_billed_cost",
      },
    ],
  },
};

interface ProcurementDashboardProps {
  id: string;
  dashboardName: string;
  tag_id?: number; // Add tag_id as optional prop
}

const ProcurementDashboard: React.FC<ProcurementDashboardProps> = ({ id, dashboardName, tag_id }) => {
  const router = useRouter();
  const [cloudPlatforms, setCloudPlatforms] = useState<Array<"azure" | "aws" | "gcp">>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTagId, setSelectedTagId] = useState<number | undefined>(undefined);
  const [selectedDuration, setSelectedDuration] = useState<string | undefined>(undefined);
  const [globalLoading, setGlobalLoading] = useState(false);

  // Helper to track how many children are loading
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
  useEffect(() => {
    if (tag_id) {
      setSelectedTagId(tag_id);
    }
  }, [tag_id]);

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

  useEffect(() => {
    const fetchDashboardDetails = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await axiosInstance.get(`${BACKEND}/dashboard/${id}`, {
          params: { dashboard_id: id },
        });
        const dashboardDetails = response.data;
        const platforms: Array<"azure" | "aws" | "gcp"> = dashboardDetails.cloud_platforms || [];
        setCloudPlatforms(platforms);
      } catch (err: any) {
        setError(err.message || "Failed to fetch dashboard details.");
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardDetails();
  }, [id]);

  if (loading) {
    return <div className="bg-gray-100 p-4 font-sans h-screen overflow-auto">Loading...</div>;
  }

  if (error) {
    return (
      <div className="bg-gray-100 p-4 font-sans h-screen overflow-auto">
        <h2 className="text-xl font-bold text-red-600">Error</h2>
        <p>{error}</p>
        <Button variant="ghost" onClick={handleBackClick}>
          Go Back
        </Button>
      </div>
    );
  }

  if (cloudPlatforms.length === 0) {
    return (
      <div className="bg-gray-100 p-4 font-sans h-screen overflow-auto">
        <h2 className="text-xl font-bold">No Cloud Platforms Selected</h2>
        <p>Please associate at least one cloud platform with this dashboard.</p>
        <Button variant="ghost" onClick={handleBackClick}>
          Go Back
        </Button>
      </div>
    );
  }

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
          <h1 className="text-3xl font-bold text-[#233E7D] tracking-tight">{dashboardName}: Procurement</h1>
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
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div className="flex-grow">
              <DashboardNameCard title="PROCUREMENT" />
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
            {cloudPlatformConfig.commonCharts.map((chart: any, index: number) => (
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
                  title={cloudPlatformConfig.commonCharts[0].title}
                  queryType={cloudPlatformConfig.commonCharts[0].queryType}
                  id={id}
                  nameField={cloudPlatformConfig.commonCharts[0].nameField}
                  valueField={cloudPlatformConfig.commonCharts[0].valueField}
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
                  title={cloudPlatformConfig.commonCharts[1].title}
                  queryType={cloudPlatformConfig.commonCharts[1].queryType}
                  id={id}
                  nameField={cloudPlatformConfig.commonCharts[1].nameField}
                  valueField={cloudPlatformConfig.commonCharts[1].valueField}
                  tag_id={selectedTagId}
                  duration={selectedDuration}
                  setParentLoading={setChildLoading}
                />
              </div>
            </div>
          </div>
        </>
      </div>
    </div>
  );
};

export default ProcurementDashboard;
