"use client";

import ExecutiveDashboard from "@/components/PersonaDashboard/ExecutiveDashboard";
import FinanceTeamDashboard from "@/components/PersonaDashboard/FinanceTeamDashboard";
import FinOpsDashboard from "@/components/PersonaDashboard/FinOpsDashboard";
import ProcurementDashboard from "@/components/PersonaDashboard/ProcurementDashboard";
import ProductOwnerDashboard from "@/components/PersonaDashboard/ProductOwnerDashboard";
import { useParams } from "next/navigation";

const PersonaDashboard = () => {
  const params = useParams();
  console.log(params)
  const dashboard_id = params.id as string;
  const dashboardName = params.dashboardName as string;
  const encodedPersona = params.persona as string;
  const persona = decodeURIComponent(encodedPersona);
  const cloud_platforms = params.cloud_platforms as "azure" | "aws" | "gcp";

  const tagIdString = Array.isArray(params.tag_id) ? params.tag_id[0] : params.tag_id;
  const tagId = tagIdString ? parseInt(tagIdString, 10) : undefined;
  console.log("tagId", tagId);

  const renderDashboard = () => {
    switch (persona) {
      case "FinOps Team":
        return (
          <FinOpsDashboard
            id={dashboard_id}
            dashboardName={dashboardName}
            tag_id={tagId}
          />
        );
      case "Executives: CEO/CTO":
        return (
          <ExecutiveDashboard
            id={dashboard_id}
            dashboardName={dashboardName}
            tag_id={tagId}        
          />
        );
      case "Product Owner":
        return (
          <ProductOwnerDashboard
            id={dashboard_id}
            dashboardName={dashboardName}
          />
        );
      case "Procurement":
        return (
          <ProcurementDashboard
            id={dashboard_id}
            dashboardName={dashboardName}
          />
        );
      case "Finance Team":
        return (
          <FinanceTeamDashboard
            id={dashboard_id}
            dashboardName={dashboardName}
          />
        );
      default:
        return <div className="text-lg font-semibold text-[#D82026]">Unknown persona: {persona}</div>;
    }
  };

  return (
    <div className="bg-[#F9FEFF] min-h-screen p-6 flex flex-col items-center">
      <h1 className="text-3xl font-bold text-[#233E7D] mb-6 w-full text-left max-w-7xl">{dashboardName}</h1>
      <div className="w-full max-w-7xl bg-white border border-[#E0E5EF] rounded-md shadow-md p-6">
        {renderDashboard()}
      </div>
    </div>
  );
};

export default PersonaDashboard;