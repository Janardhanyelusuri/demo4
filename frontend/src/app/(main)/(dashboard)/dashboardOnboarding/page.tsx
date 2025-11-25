"use client";
import React, { useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import AWS from "@/assets/AWS.svg";
import GOOGLE from "@/assets/google.svg";
import AZURE from "@/assets/azure.svg";
import SNOWFLAKE from "@/assets/snowflake.svg";
import Image from "next/image";
import DashboardOnboardingForm from "@/components/DashboardOnboarding/DashboardOnboardingForm";
import Link from "next/link"; 
type Props = {};

const DashboardOnboardingContent = () => {
  const searchParams = useSearchParams();
  const projectName = searchParams.get("dashboardName");
  const [selectedProviders, setSelectedProviders] = useState<string[]>([]);
  const [isReadyToProceed, setIsReadyToProceed] = useState(false);

  const handleProviderClick = (provider: string) => {
    setSelectedProviders((prev) => 
      prev.includes(provider)
        ? prev.filter((p) => p !== provider)
        : [...prev, provider]
    );
  };

  const handleProceed = () => {
    if (selectedProviders.length > 0) {
      setIsReadyToProceed(true);
    }
  };

  const handleBackClick = () => {
    setSelectedProviders([]);
    setIsReadyToProceed(false);
  };

  return (
  <div className="bg-[#F9FEFF] min-h-screen p-4">
    {/* <ProjectBreadcrumb projectName={projectName} /> */}
    {isReadyToProceed ? (
      <DashboardOnboardingForm
        projectName={projectName}
        cloud_platforms={selectedProviders}
        onBack={handleBackClick}
      />
    ) : (
      <div className="flex flex-col justify-center items-center h-screen -mt-10">
        <div className="text-4xl font-bold mb-4 text-[#1A1D23]">Let&apos;s start</div>
        <div className="w-[450px] shadow-lg border border-[#D1D5DB] rounded-md overflow-hidden bg-white">
          <div className="bg-[#F7F7F7] p-2 text-sm text-[#1A1D23] font-semibold border-b border-[#D1D5DB]">
            Select Options
          </div>
          <div className="bg-white p-6">
            <div className="text-lg font-semibold mb-2 text-[#1A1D23]">Cloud Providers</div>
            <div className="flex gap-10 justify-around mb-6">
              {[
                { name: "AWS", src: AWS },
                { name: "Azure", src: AZURE },
                { name: "GCP", src: GOOGLE },
              ].map((provider) => (
                <div 
                  key={provider.name}
                  className="relative flex flex-col items-center"
                >
                  <button
                    className={`
                      flex flex-col items-center justify-center 
                      w-[100px] p-4 bg-white shadow-md rounded-md 
                      cursor-pointer hover:shadow-lg hover:border-[#1A1D23] border transition-all duration-300 ease-in-out
                      relative
                      ${selectedProviders.includes(provider.name) 
                        ? "border-2 border-[#1A1D23] bg-[#F7F7F7]" 
                        : "border-2 border-[#D1D5DB]"}
                    `}
                    onClick={() => handleProviderClick(provider.name)}
                  >
                    <Image 
                      src={provider.src} 
                      alt={provider.name} 
                      width={54} 
                      height={54} 
                    />
                    <span className="mt-2 text-sm text-[#1A1D23] font-medium">{provider.name}</span>
                  </button>
                  <div 
                    className={`
                      absolute top-0 right-0 w-5 h-5 
                      border-2 rounded 
                      ${selectedProviders.includes(provider.name) 
                        ? 'bg-[#1A1D23] border-[#1A1D23]' 
                        : 'bg-white border-[#D1D5DB]'}
                    `}
                  >
                    {selectedProviders.includes(provider.name) && (
                      <svg 
                        xmlns="http://www.w3.org/2000/svg" 
                        viewBox="0 0 24 24" 
                        fill="none" 
                        stroke="white" 
                        strokeWidth="3" 
                        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-4 h-4"
                      >
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                    )}
                  </div>
                </div>
              ))}
            </div>
            
            <div className="text-lg font-semibold mb-2 text-[#1A1D23]">Warehouse</div>
            <div className="flex gap-10 justify-around mb-6">
              <button
                disabled={true}
                className="disabled:opacity-50 flex flex-col items-center justify-center w-[100px] p-4 bg-white shadow-md rounded-md border border-[#D1D5DB] cursor-not-allowed"
              >
                <Image src={SNOWFLAKE} alt="Snowflake" width={54} height={54} />
                <span className="mt-2 text-sm text-[#6B7280] font-medium">Snowflake</span>
              </button>
            </div>

            {selectedProviders.length > 0 && (
              <div className="flex justify-center mt-4">
                <button
                  onClick={handleProceed}
                  className="px-6 py-2 bg-[#4F46E5] text-white rounded-md hover:bg-[#4338CA] transition-colors text-sm font-medium h-10"
                >
                  Proceed
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    )}
  </div>
);
};

const DashboardOnboarding = (props: Props) => {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <DashboardOnboardingContent />
    </Suspense>
  );
};

export default DashboardOnboarding;