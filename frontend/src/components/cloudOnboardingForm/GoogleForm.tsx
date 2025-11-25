"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import GCP from "@/assets/google.svg";
import { Unplug, CalendarIcon } from "lucide-react";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  format,
  subMonths,
  subDays,
  isAfter,
  isBefore,
  startOfDay,
} from "date-fns";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Calendar } from "@/components/ui/calendar";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/use-toast";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import Image from "next/image";
import axiosInstance, { BACKEND, createProject, submitGcpFormData, testGcpConnection } from "@/lib/api";


interface GoogleFormProps {
  projectName: string | null;
  providerName: string;
  onBack: () => void;
}

interface GCPProject {
  project_id: string;
  project_name: string;
}

const formSchema = z.object({
  credentialFile: z
    .instanceof(File, { message: "Credential File is required." })
    .refine((file) => file.size > 0, { message: "File cannot be empty." }),
  gcpProject: z.string().min(1, { message: "GCP Project is required." }),
  fromDate: z.date({
    required_error: "From date is required.",
  }),
  yearlyBudget: z
    .string()
    .regex(/^\d+$/, { message: "Enter a valid number." })
    .optional(),
  export: z.boolean().default(true),
  datasetId: z.string().min(1, { message: "Dataset ID is required." }),
  billingAccountId: z
    .string()
    .min(1, { message: "Billing Account ID is required." }),
});

const GoogleForm: React.FC<GoogleFormProps> = ({
  projectName,
  providerName,
  onBack,
}) => {
  const { toast } = useToast();
  const router = useRouter();
  const [connectionStatus, setConnectionStatus] = useState<
    "idle" | "success" | "failed"
  >("idle");
  const [gcpProjects, setGcpProjects] = useState<GCPProject[]>([]);
  const today = startOfDay(new Date());
  const yesterday = subDays(today, 1);
  const twelveMonthsAgo = subMonths(yesterday, 12);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      credentialFile: undefined,
      gcpProject: "",
      fromDate: yesterday,
      yearlyBudget: "",
      export: true,
      datasetId: "",
      billingAccountId: "",
    },
  });

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    const selectedProject = gcpProjects.find(
      (project) => project.project_id === values.gcpProject
    );
    if (!selectedProject) {
      toast({
        title: "Error",
        description: "Please select a valid GCP project.",
        variant: "destructive",
      });
      return;
    }

    try {
      const formData = new FormData();
      formData.append("file", values.credentialFile);

      const testData = await testGcpConnection(formData);

      if (!testData.status) {
        toast({
          title: "Connection Failed",
          description:
            testData.message ||
            "Failed to connect to GCP. Please check your credentials.",
          variant: "destructive",
        });
        return;
      }

      const projectRequestBody = {
        name: projectName || "Unnamed Project",
        status: false,
        date: new Date().toISOString().split("T")[0],
        cloud_platform: providerName.toLowerCase(),
      };

      const projectData = await createProject(projectRequestBody);
      const projectId = projectData.id;

      const gcpRequestBody = {
        credential_file: values.credentialFile,
        date: format(values.fromDate, "yyyy-MM-dd"),
        project_id: projectId,
        gcp_project_id: selectedProject.project_id,
        gcp_project_name: selectedProject.project_name,
        yearly_budget: values.yearlyBudget,
        export: values.export,
        status: false,
        dataset_id: values.datasetId,
        billing_account_id: values.billingAccountId,
      };

      await submitGcpFormData(gcpRequestBody);

      toast({
        title: "Success",
        description: "Form submitted successfully.",
      });
      router.push("/connections");
    } catch (error) {
      console.error("Error submitting form:", error);
      toast({
        title: "Error",
        description: "An unexpected error occurred. Please try again later.",
        variant: "destructive",
      });
    }
  };

  const testConnection = async () => {
    const credentialFile = form.getValues("credentialFile");
    if (!credentialFile) {
      toast({
        title: "Error",
        description: "Please upload a credential file first.",
        variant: "destructive",
      });
      return;
    }

    const formData = new FormData();
    formData.append("file", credentialFile);

    try {
      const data = await testGcpConnection(formData);

      if (data.status) {
        setConnectionStatus("success");
        fetchGCPProjects(formData);
      } else {
        setConnectionStatus("failed");
        toast({
          title: "Connection Failed",
          description: data.message || "Failed to connect to GCP.",
          variant: "destructive",
        });
      }
    } catch (error) {
      setConnectionStatus("failed");
      toast({
        title: "Error",
        description: "An error occurred while testing the connection.",
        variant: "destructive",
      });
    }
  };

  const fetchGCPProjects = async (formData: FormData) => {
    try {
      const response = await axiosInstance.post(`${BACKEND}/gcp/get_projects`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      const data = response.data;
      setGcpProjects(data.projects);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch GCP projects.",
        variant: "destructive",
      });
      console.error("Error fetching GCP projects:", error);
    }
  };

  const getTestConnectionButtonVariant = () => {
    switch (connectionStatus) {
      case "success":
        return "secondary";
      case "failed":
        return "destructive";
      default:
        return "outline";
    }
  };

  return (
    <div className="shadow-lg border border-[#E0E5EF] rounded-md h-[800px] mb-9 bg-white">
      <div className="flex items-center gap-4 w-[600px] bg-[#EDEFF2] pl-2 mb-6 rounded-t-md">
        <Image
          className="mt-1 p-2"
          src={GCP}
          alt="GCP"
          width={50}
          height={50}
        />
        <p className="text-lg font-semibold text-[#233E7D]">{projectName}</p>
      </div>
      <div className="p-4 rounded-b-md overflow-auto bg-white h-[750px]">
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
            <FormField
              control={form.control}
              name="credentialFile"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Credential File</FormLabel>
                  <FormControl>
                    <Input
                      id="credentialFile"
                      type="file"
                      className="border-[#C8C8C8] rounded-md px-3 py-2 text-base"
                      onChange={(e) =>
                        field.onChange(
                          e.target.files ? e.target.files[0] : null
                        )
                      }
                    />
                  </FormControl>
                  <FormMessage className="text-xs text-[#D82026]" />
                </FormItem>
              )}
            />
            <Button
              type="button"
              variant={getTestConnectionButtonVariant()}
              onClick={testConnection}
              className="bg-[#D82026] text-white rounded-md hover:bg-[#b81a1f] text-sm font-medium h-10 px-4 py-2"
            >
              <Unplug />
              <span className="ml-2">Test Connection</span>
            </Button>
            <FormField
              control={form.control}
              name="gcpProject"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">GCP Project</FormLabel>
                  <FormControl>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" className="border-[#C8C8C8] rounded-md">
                          {field.value
                            ? gcpProjects.find((p) => p.project_id === field.value)?.project_name || "Select GCP Project"
                            : "Select GCP Project"}
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="w-56">
                        <DropdownMenuGroup>
                          {gcpProjects.map((project) => (
                            <DropdownMenuItem
                              key={project.project_id}
                              onClick={() => field.onChange(project.project_id)}
                            >
                              {project.project_name}
                            </DropdownMenuItem>
                          ))}
                        </DropdownMenuGroup>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </FormControl>
                  <FormMessage className="text-xs text-[#D82026]" />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="fromDate"
              render={({ field }) => (
                <FormItem className="flex flex-col">
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">From</FormLabel>
                  <Popover>
                    <PopoverTrigger asChild>
                      <FormControl>
                        <Button
                          variant={"outline"}
                          className={cn(
                            "w-[240px] pl-3 text-left font-normal border-[#C8C8C8] rounded-md",
                            !field.value && "text-muted-foreground"
                          )}
                        >
                          {field.value ? (
                            format(field.value, "PPP")
                          ) : (
                            <span>Pick a date</span>
                          )}
                          <CalendarIcon className="ml-auto h-4 w-4 opacity-50" />
                        </Button>
                      </FormControl>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start">
                      <Calendar
                        mode="single"
                        selected={field.value}
                        onSelect={field.onChange}
                        defaultMonth={yesterday}
                        disabled={(date) =>
                          isAfter(date, yesterday) ||
                          isBefore(date, twelveMonthsAgo)
                        }
                        initialFocus
                      />
                    </PopoverContent>
                  </Popover>
                  <FormMessage className="text-xs text-[#D82026]" />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="yearlyBudget"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Yearly Budget ($)</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      min="0"
                      placeholder="Enter Yearly Budget"
                      className="border-[#C8C8C8] rounded-md px-3 py-2 text-base"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage className="text-xs text-[#D82026]" />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="export"
              render={({ field }) => (
                <FormItem className="flex flex-row items-start space-x-3 space-y-0 rounded-md border border-[#E0E5EF] p-4">
                  <FormControl>
                    <Checkbox
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                  <div className="space-y-1 leading-none">
                    <FormLabel className="text-sm font-semibold text-[#233E7D]">Export</FormLabel>
                  </div>
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="datasetId"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Dataset Name</FormLabel>
                  <FormControl>
                    <Input type="text" placeholder="Dataset Name" className="border-[#C8C8C8] rounded-md px-3 py-2 text-base" {...field} />
                  </FormControl>
                  <FormMessage className="text-xs text-[#D82026]" />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="billingAccountId"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Billing Account ID</FormLabel>
                  <FormControl>
                    <Input
                      type="text"
                      placeholder="Enter Billing Account ID"
                      className="border-[#C8C8C8] rounded-md px-3 py-2 text-base"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage className="text-xs text-[#D82026]" />
                </FormItem>
              )}
            />

            <div className="flex gap-4 mt-4">
              <button
                onClick={onBack}
                type="button"
                className="flex items-center px-4 py-2 border border-[#233E7D] rounded-md text-[#233E7D] hover:bg-[#233E7D]/10 text-sm font-medium"
              >
                Back
              </button>
              <Button
                variant="default"
                type="submit"
                className="bg-[#D82026] text-white rounded-md hover:bg-[#b81a1f] text-sm font-medium h-10 px-4 py-2"
              >
                Confirm
              </Button>
            </div>
          </form>
        </Form>
      </div>
    </div>
  );
};

export default GoogleForm;
