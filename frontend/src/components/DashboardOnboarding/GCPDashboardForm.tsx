import React, { useState, useEffect } from "react";
import Image from "next/image";
import google from "@/assets/google.svg";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import Select, { components } from "react-select";
import axiosInstance, { BACKEND } from "@/lib/api";

interface Project {
  id: number;
  name: string;
  status: boolean;
  date: string;
  cloud_platform: string;
}

interface GCPDashboardFormProps {
  projectName: string | null;
  providerName: string;
  onBack: () => void;
}

const GCPDashboardForm: React.FC<GCPDashboardFormProps> = ({
  projectName,
  providerName,
  onBack,
}) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const router = useRouter();
  const [filteredProjects, setFilteredProjects] = useState<Project[]>([]);
  const [selectedProjectIds, setSelectedProjectIds] = useState<number[]>([]);
  const [selectedPersonas, setSelectedPersonas] = useState<string[]>([]);

  const personas = [
    { value: "FinOps Team", label: "FinOps Team" },
    { value: "Executives: CEO/CTO", label: "Executives: CEO/CTO" },
    { value: "Product Owner", label: "Product Owner" },
  ];

  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const response = await axiosInstance.get(`${BACKEND}/project/`);
        setProjects(response.data);
      } catch (error) {
        console.error("Error fetching projects:", error);
      }
    };
    fetchProjects();
  }, []);

  useEffect(() => {
    const filtered = projects.filter(
      (project) => project.cloud_platform === providerName.toLowerCase()
    );
    setFilteredProjects(filtered);
  }, [projects, providerName]);

  const handlePersonaChange = (selectedOptions: any) => {
    setSelectedPersonas(
      selectedOptions ? selectedOptions.map((option: any) => option.value) : []
    );
  };

  const handleProjectChange = (selectedOptions: any) => {
    setSelectedProjectIds(
      selectedOptions ? selectedOptions.map((option: any) => option.value) : []
    );
  };

  const onSubmit = async () => {
    if (selectedProjectIds.length === 0) {
      alert("Please select at least one project");
      return;
    }

    try {
      const response = await axiosInstance.post(
        `${BACKEND}/dashboard/`,
        {
          name: projectName,
          status: true,
          date: new Date().toISOString().split("T")[0],
          cloud_platform: providerName.toLowerCase(),
          persona: selectedPersonas,
          project_ids: selectedProjectIds,
        },
        {
          headers: {
            accept: "application/json",
            "Content-Type": "application/json",
          },
        }
      );

      console.log("Dashboard created:", response.data);
      router.push("/dashboard-home");
    } catch (error) {
      console.error("Error creating dashboard:", error);
      alert("Failed to create dashboard. Please try again.");
    }
  };

  const CustomOption = (props: any) => (
    <components.Option {...props}>
      <input
        type="checkbox"
        checked={props.isSelected}
        onChange={() => null}
        style={{ marginRight: 8 }}
      />
      <label>{props.label}</label>
    </components.Option>
  );

  return (
    <div className="shadow-md border border-[#E0E5EF] rounded-md h-[550px] mb-9 bg-white flex flex-col">
      <div className="flex items-center gap-4 w-[600px] bg-[#EDEFF2] pl-2 mb-6 rounded-t-md">
        <Image className="mt-1 p-2" src={google} alt="GCP" width={50} height={50} />
        <p className="text-lg font-semibold text-[#233E7D]">{projectName}</p>
      </div>
      <div className="flex-1 bg-white p-4 flex flex-col justify-between">
        <div className="w-[550px]">
          <label
            htmlFor="project-dropdown"
            className="block mb-2 text-sm font-semibold text-[#233E7D]"
          >
            Select Provider Connections:
          </label>
          <Select
            id="project-dropdown"
            isMulti
            options={filteredProjects.map((project) => ({
              value: project.id,
              label: project.name,
            }))}
            className="mb-4"
            onChange={handleProjectChange}
            components={{ Option: CustomOption }}
            closeMenuOnSelect={false}
            hideSelectedOptions={false}
          />

          <label
            htmlFor="persona-dropdown"
            className="block mb-2 text-sm font-semibold text-[#233E7D]"
          >
            Select Personas:
          </label>
          <Select
            id="persona-dropdown"
            isMulti
            options={personas}
            className="mb-4"
            onChange={handlePersonaChange}
            components={{ Option: CustomOption }}
            closeMenuOnSelect={false}
            hideSelectedOptions={false}
          />
        </div>
        <div className="flex gap-4 mt-4">
          <button
            onClick={onBack}
            className="flex items-center px-4 py-2 border border-[#233E7D] rounded-md text-[#233E7D] hover:bg-[#233E7D]/10 text-sm font-medium transition-colors"
          >
            Back
          </button>
          <Button
            className="bg-[#D82026] text-white rounded-md hover:bg-[#b81a1f] text-sm font-medium h-10 px-4 py-2"
            type="submit"
            onClick={onSubmit}
          >
            Confirm
          </Button>
        </div>
      </div>
    </div>
  );
};

export default GCPDashboardForm;
