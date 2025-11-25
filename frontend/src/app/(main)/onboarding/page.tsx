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
import CloudOnboardingForm from "@/components/cloudOnboardingForm/CloudOnboardingForm";
import { Trash } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import Link from "next/link"; 

type Props = {};

const OnboardingPageContent = () => {
  const searchParams = useSearchParams();
  const projectName = searchParams.get("projectName");
  const platformFromUrl = searchParams.get("platform");
  const [selectedProvider, setSelectedProvider] = useState<string | null>(platformFromUrl);
  const { toast } = useToast();

  const handleProviderClick = (provider: string) => {
    setSelectedProvider(provider);
  };

  const handleBackClick = () => {
    setSelectedProvider(null);
  };

  return (
    <div className="p-4 bg-[#F9FEFF] min-h-screen">
      {/* <ProjectBreadcrumb projectName={projectName} /> */}
      {selectedProvider ? (
        <CloudOnboardingForm
          projectName={projectName}
          providerName={selectedProvider}
          onBack={handleBackClick}
        />
      ) : (
        <div className="flex flex-col justify-center items-center h-screen -mt-10">
          <div className="text-3xl font-bold text-[#233E7D] mb-4">Let&apos;s start</div>
          <div className="w-[450px] shadow-lg border border-[#E0E5EF] rounded-md overflow-hidden bg-white">
            <div className="bg-[#EDEFF2] p-2 text-sm text-[#233E7D] font-semibold border-b border-[#E0E5EF]">
              Select Options
            </div>
            <div className="bg-white p-6">
              <div className="text-lg font-semibold text-[#233E7D] mb-2">Cloud Providers</div>
              <div className="flex gap-6 justify-around mb-8">
                <button
                  className="flex flex-col items-center justify-center w-1/4 p-4 bg-white shadow-md border border-[#E0E5EF] rounded-md cursor-pointer hover:shadow-lg hover:border-[#233E7D] transition-all duration-200"
                  onClick={() => handleProviderClick("AWS")}
                >
                  <Image src={AWS} alt="AWS" width={54} height={54} />
                  <span className="mt-2 text-sm text-[#233E7D] font-medium">AWS</span>
                </button>
                <button
                  className="flex flex-col items-center justify-center w-1/4 p-4 bg-white shadow-md border border-[#E0E5EF] rounded-md cursor-pointer hover:shadow-lg hover:border-[#233E7D] transition-all duration-200"
                  onClick={() => handleProviderClick("Azure")}
                >
                  <Image src={AZURE} alt="Azure" width={54} height={54} />
                  <span className="mt-2 text-sm text-[#233E7D] font-medium">Azure</span>
                </button>
                <button
                  className="flex flex-col items-center justify-center w-1/4 p-4 bg-white shadow-md border border-[#E0E5EF] rounded-md cursor-pointer hover:shadow-lg hover:border-[#233E7D] transition-all duration-200"
                  onClick={() => handleProviderClick("GCP")}
                >
                  <Image src={GOOGLE} alt="GCP" width={54} height={54} />
                  <span className="mt-2 text-sm text-[#233E7D] font-medium">GCP</span>
                </button>
              </div>
              <div className="text-lg font-semibold text-[#233E7D] mb-2">Warehouse</div>
              <div className="flex gap-6 justify-around mb-2">
                <button
                  disabled={true}
                  className="disabled:opacity-50 flex flex-col items-center justify-center w-1/4 p-4 bg-white shadow-md border border-[#E0E5EF] rounded-md cursor-not-allowed"
                  onClick={() => handleProviderClick("Snowflake")}
                >
                  <Image
                    src={SNOWFLAKE}
                    alt="Snowflake"
                    width={54}
                    height={54}
                  />
                  <span className="mt-2 text-sm text-[#6B7280] font-medium">Snowflake</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const OnboardingPage = (props: Props) => {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <OnboardingPageContent />
    </Suspense>
  );
};

export default OnboardingPage;