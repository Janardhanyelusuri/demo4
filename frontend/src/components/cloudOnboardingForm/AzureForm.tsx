"use client";

import React, { useState } from "react";
import Image from "next/image";
import AZURE from "@/assets/azure.svg";
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
import { useRouter } from "next/navigation";
import axiosInstance, { BACKEND } from "@/lib/api";


interface AzureFormProps {
  projectName: string | null;
  providerName: string;
  onBack: () => void;
}

const formSchema = z.object({
  tenantId: z.string().min(1, { message: "Tenant ID is required." }),
  applicationId: z.string().min(1, { message: "Application ID is required." }),
  clientSecret: z.string().min(1, { message: "Client Secret is required." }),
  subscription: z.object({
    subscription_id: z.string(),
    display_name: z.string(),
  }),
  fromDate: z.date({
    required_error: "From date is required.",
  }),
  yearlyBudget: z
    .string()
    .regex(/^\d+$/, { message: "Enter a valid number." })
    .optional(),
  storageAccountName: z
    .string()
    .min(1, { message: "Storage Account Name is required." }),
  export: z.boolean().default(true),
  containerName: z.string().optional(),
  resourceGroupName: z.string().optional(),
});

const AzureForm: React.FC<AzureFormProps> = ({ projectName, providerName, onBack }) => {
  const { toast } = useToast();
  const router = useRouter();
  const [subscriptions, setSubscriptions] = useState<any[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<"idle" | "success" | "failed">("idle");

  const today = startOfDay(new Date());
  const yesterday = subDays(today, 1);
  const twelveMonthsAgo = subMonths(yesterday, 12);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      tenantId: "",
      applicationId: "",
      clientSecret: "",
      subscription: { subscription_id: "", display_name: "" },
      fromDate: yesterday,
      yearlyBudget: "",
      storageAccountName: "",
      export: true,
      containerName: "",
      resourceGroupName: "",
    },
  });

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      const testResponse = await axiosInstance.post(`${BACKEND}/azure/test_connection`, {
        azure_tenant_id: values.tenantId,
        azure_client_id: values.applicationId,
        azure_client_secret: values.clientSecret,
      }, {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      });

      const testData = testResponse.data;

      if (!testData.status) {
        toast({
          title: "Connection Failed",
          description: testData.message || "Failed to connect to Azure. Please check your credentials.",
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

      const projectResponse = await axiosInstance.post(`${BACKEND}/project/`, projectRequestBody, {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      });

      const projectData = projectResponse.data;
      const projectId = projectData.id;

      const azurePayload = {
        azure_tenant_id: values.tenantId,
        azure_client_id: values.applicationId,
        azure_client_secret: values.clientSecret,
        subscription_info: values.subscription,
        date: format(values.fromDate, "yyyy-MM-dd"),
        project_id: projectId,
        yearly_budget: values.yearlyBudget,
        monthly_budget: values.yearlyBudget,
        quarterly_budget: values.yearlyBudget,
        storage_account_name: values.storageAccountName,
        export: values.export,
        container_name: values.export ? values.containerName : undefined,
        resource_group_name: !values.export ? values.resourceGroupName : undefined,
        status: false,
      };

      const azureResponse = await axiosInstance.post(`${BACKEND}/azure/`, azurePayload, {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      });

      if (azureResponse.status === 200) {
        toast({
          title: "Submission Successful",
          description: "Your data has been successfully submitted.",
        });
        router.push("/connections");
      } else {
        throw new Error("Azure API call failed");
      }
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
    const { tenantId, applicationId, clientSecret } = form.getValues();

    if (!tenantId || !applicationId || !clientSecret) {
      toast({
        title: "Error",
        description: "Please fill in all required fields before testing the connection.",
        variant: "destructive",
      });
      return;
    }

    try {
      const response = await axiosInstance.post(`${BACKEND}/azure/test_connection`, {
        azure_tenant_id: tenantId,
        azure_client_id: applicationId,
        azure_client_secret: clientSecret,
      }, {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      });

      const data = response.data;

      if (data.status) {
        setConnectionStatus("success");
        toast({
          title: "Connection Successful",
          description: data.message,
        });

        await fetchSubscriptions({
          azure_tenant_id: tenantId,
          azure_client_id: applicationId,
          azure_client_secret: clientSecret,
        });
      } else {
        setConnectionStatus("failed");
        toast({
          title: "Connection Failed",
          description: data.message || "An error occurred while testing the connection.",
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

  const fetchSubscriptions = async (credentials: { azure_tenant_id: string; azure_client_id: string; azure_client_secret: string; }) => {
    try {
      const response = await axiosInstance.post(`${BACKEND}/azure/get_subscriptions`, credentials, {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      });

      const data = response.data;

      if (data.subscriptions) {
        setSubscriptions(data.subscriptions);
      } else {
        toast({
          title: "Error",
          description: "Failed to fetch subscriptions.",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch subscriptions.",
        variant: "destructive",
      });
    }
  };

  const getTestConnectionButtonVariant = () => {
    switch (connectionStatus) {
      case "success":
        return "default";
      case "failed":
        return "destructive";
      default:
        return "outline";
    }
  };

  return (
    <div className="shadow-lg border border-[#E0E5EF] rounded-md h-[800px] mb-9 bg-white">
      <div className="flex items-center gap-4 w-[600px] bg-[#EDEFF2] pl-2 mb-6 rounded-t-md">
        <Image className="mt-1 p-2" src={AZURE} alt="Azure" width={50} height={50} />
        <p className="text-lg font-semibold text-[#233E7D]">{projectName}</p>
      </div>
      <div className="p-4 rounded-b-md overflow-auto bg-white h-[750px]">
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="tenantId"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Tenant ID</FormLabel>
                  <FormControl>
                    <Input
                      type="text"
                      placeholder="Enter Tenant ID"
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
              name="applicationId"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Application ID</FormLabel>
                  <FormControl>
                    <Input
                      type="text"
                      placeholder="Enter Application ID"
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
              name="clientSecret"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Client Secret</FormLabel>
                  <FormControl>
                    <Input
                      type="password"
                      placeholder="Enter Client Secret"
                      className="border-[#C8C8C8] rounded-md px-3 py-2 text-base"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage className="text-xs text-[#D82026]" />
                </FormItem>
              )}
            />
            <Button
              variant={getTestConnectionButtonVariant()}
              onClick={(event) => {
                event.preventDefault();
                testConnection();
              }}
              className="bg-[#D82026] text-white rounded-md hover:bg-[#b81a1f] text-sm font-medium h-10 px-4 py-2"
            >
              <Unplug />
              <span className="ml-2">Test Connection</span>
            </Button>
            <FormField
              control={form.control}
              name="subscription"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Subscription</FormLabel>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" className="w-full justify-between border-[#C8C8C8] rounded-md">
                        {field.value.display_name || "Select Subscription"}
                        <Unplug className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent className="w-72">
                      <DropdownMenuGroup>
                        {subscriptions.map((subscription) => (
                          <DropdownMenuItem
                            key={subscription.subscription_id}
                            onClick={() => {
                              field.onChange(subscription);
                            }}
                          >
                            {subscription.display_name}
                          </DropdownMenuItem>
                        ))}
                      </DropdownMenuGroup>
                    </DropdownMenuContent>
                  </DropdownMenu>
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
                        <Button variant={"outline"} className={cn(
                          "w-[240px] pl-3 text-left font-normal border-[#C8C8C8] rounded-md",
                          !field.value && "text-muted-foreground"
                        )}>
                          {field.value ? format(field.value, "PPP") : <span>Pick a date</span>}
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
                        disabled={(date) => isAfter(date, yesterday) || isBefore(date, twelveMonthsAgo)}
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
                    <Input type="number" min="0" placeholder="Enter Yearly Budget" className="border-[#C8C8C8] rounded-md px-3 py-2 text-base" {...field} />
                  </FormControl>
                  <FormMessage className="text-xs text-[#D82026]" />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="storageAccountName"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Storage Account Name</FormLabel>
                  <FormControl>
                    <Input
                      type="text"
                      placeholder="Enter Storage Account Name"
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
            {form.watch("export") ? (
              <FormField
                control={form.control}
                name="containerName"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-sm font-semibold text-[#233E7D]">Container Name</FormLabel>
                    <FormControl>
                      <Input
                        type="text"
                        placeholder="Enter Container Name"
                        className="border-[#C8C8C8] rounded-md px-3 py-2 text-base"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage className="text-xs text-[#D82026]" />
                  </FormItem>
                )}
              />
            ) : (
              <FormField
                control={form.control}
                name="resourceGroupName"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-sm font-semibold text-[#233E7D]">Resource Group Name</FormLabel>
                    <FormControl>
                      <Input
                        type="text"
                        placeholder="Enter Resource Group Name"
                        className="border-[#C8C8C8] rounded-md px-3 py-2 text-base"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage className="text-xs text-[#D82026]" />
                  </FormItem>
                )}
              />
            )}

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

export default AzureForm;
