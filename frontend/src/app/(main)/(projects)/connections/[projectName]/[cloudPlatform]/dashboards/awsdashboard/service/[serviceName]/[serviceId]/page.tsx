"use client"
import React from "react";
import { useParams } from "next/navigation";
import DashboardAccess from "@/components/request/DashboardAccess";

const ServicePage = () => {
  const params = useParams();
  const serviceName = params.serviceName as string;
  const serviceId = params.serviceId as string;
  const projectId = params.projectName as string;

  return (
    <div>
      <h1>
        <DashboardAccess
          serviceName={serviceName}
          serviceId={serviceId}
          projectId={projectId}
        />
      </h1>
    </div>
  );
};

export default ServicePage;
