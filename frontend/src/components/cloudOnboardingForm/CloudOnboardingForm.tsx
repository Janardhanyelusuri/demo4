import React from "react";
import AwsForm from "./AwsForm";
import AzureForm from "./AzureForm";
import GoogleForm from "./GoogleForm";
import SnowflakeForm from "./SnowflakeForm";
import { CircleArrowLeft } from "lucide-react";

type CloudOnboardingFormProps = {
  projectName: string | null;
  providerName: string;
  onBack: () => void;
};

const CloudOnboardingForm: React.FC<CloudOnboardingFormProps> = ({
  projectName,
  providerName,
  onBack,
}) => {
  const renderForm = () => {
    switch (providerName) {
      case "AWS":
        return (
          <AwsForm projectName={projectName} providerName={providerName} onBack={onBack} />
        );
      case "Azure":
        return (
          <AzureForm projectName={projectName} providerName={providerName} onBack={onBack} />
        );
      case "GCP":
        return (
          <GoogleForm projectName={projectName} providerName={providerName} onBack={onBack} />
        );
      case "Snowflake":
        return (
          <SnowflakeForm projectName={projectName} providerName={providerName} />
        );
      default:
        return <div>No provider selected</div>;
    }
  };

  return (
    <div className="flex flex-col justify-center items-center overflow-hidden">
      <button
        onClick={onBack}
        className="mb-4 flex items-center text-[#233E7D] hover:text-[#D82026] text-base font-medium"
      >
        <CircleArrowLeft className="mr-2" />
        Back
      </button>
      {renderForm()}
    </div>
  );
};

export default CloudOnboardingForm;
