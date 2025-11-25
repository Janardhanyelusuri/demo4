"use client";

import React, { useState } from "react";
import Snowflake from '@/assets/snowflake.svg'
import { Unplug, CalendarIcon } from "lucide-react";
import { zodResolver } from "@hookform/resolvers/zod";
import { format } from "date-fns";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
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
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuSeparator, DropdownMenuGroup, DropdownMenuItem , DropdownMenuLabel} from "@/components/ui/dropdown-menu";
import { useToast } from "@/components/ui/use-toast";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { BACKEND } from "@/lib/api";

interface SnowflakeFormProps {
  projectName: string | null;
  providerName: string;
}

const formSchema = z.object({
  accountName: z.string().min(1, { message: "Account name is required." }),
  username: z.string().min(1, { message: "Username is required." }),
  password: z.string().min(1, { message: "Password is required." }),
  warehouse: z.object({
    name: z.string(),
    state: z.string(),
  }),
  fromDate: z.date({
    required_error: "From date is required.",
  }),
});

const SnowflakeForm: React.FC<SnowflakeFormProps> = ({ projectName, providerName }) => {
  const { toast } = useToast();
  const router = useRouter();
  const [warehouse, setWarehouse] =useState<any[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<
    "idle" | "success" | "failed"
  >("idle");

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      accountName: "",
      username: "",
      password: "",
      warehouse: { name: "", state: "" },
      fromDate: new Date(),
    },
  });

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    // First, test the connection
    const { accountName, username, password, warehouse } = values;
  
    try {
      const testResponse = await fetch(
        `${BACKEND}/snowflake/test_connection`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            accept: "application/json",
          },
          body: JSON.stringify({
            account_name: accountName,
            user_name: username,
            password: password,
          }),
        }
      );
  
      const testData = await testResponse.json();
  
      if (!testData.status) {
        toast({
          title: "Connection Failed",
          description:
            testData.message ||
            "Failed to connect to Snowflake. Please check your credentials.",
          variant: "destructive",
        });
        return; // Stop here if the connection test fails
      }
  
      // If connection test passes, proceed with project creation and data submission
      const projectRequestBody = {
        name: projectName || "Unnamed Project",
        status: true,
        date: new Date().toISOString().split("T")[0],
        cloud_platform: providerName.toLowerCase(),
      };
  
      const projectResponse = await fetch(`${BACKEND}/project/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
        body: JSON.stringify(projectRequestBody),
      });
  
      if (!projectResponse.ok) {
        const errorText = await projectResponse.text();
        if (projectResponse.status === 409) {
          toast({
            title: "Error",
            description: "Project name already taken",
            variant: "destructive",
          });
          return;
        }
        throw new Error(`Failed to create project: ${errorText}`);
      }
  
      const projectData = await projectResponse.json();
      const projectId = projectData.id;
  
      // Now, submit the Snowflake form data
      const snowflakeRequestBody = {
        account_name: accountName,
        user_name: username,
        password: password,
        warehouse_name: warehouse.name, // Only sending the warehouse name
        date: format(values.fromDate, "yyyy-MM-dd"),
        project_id: projectId,
      };
  
      const snowflakeResponse = await fetch(`${BACKEND}/snowflake/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
        body: JSON.stringify(snowflakeRequestBody),
      });
  
      if (snowflakeResponse.ok) {
        toast({
          title: "Success",
          description: "Form submitted successfully.",
        });
        router.push("/connections");
      } else {
        throw new Error("Snowflake API call failed");
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
    const { accountName, username, password} = form.getValues();
    if (!accountName || !username || !password) {
      toast({
        title: "Error",
        description: "Please enter all credentials.",
        variant: "destructive",
      });
      return;
    }

    try {
      const response = await fetch(
        `${BACKEND}/snowflake/test_connection`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            accept: "application/json",
          },
          body: JSON.stringify({
            account_name: accountName,
            user_name: username,
            password: password,
          }),
        }
      );

      const data = await response.json();
      console.log(data);

      if (data.status) {
        setConnectionStatus("success");
        toast({
          title: "Connection Successful",
          description: data.message,
        });
        // Fetch subscriptions after successful connection
        await fetchWarehouse({
          account_name: accountName,
          user_name: username,
          password: password,
        });
      } else {
        setConnectionStatus("failed");
        toast({
          title: "Connection Failed",
          description: data.message || "Failed to connect to Snowflake.",
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

  const fetchWarehouse = async (credentials: {
    account_name: string;
    user_name: string;
    password: string;
  }) => {
    try {
      const response = await fetch(
        `${BACKEND}/snowflake/get_warehouses`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            accept: "application/json",
          },
          body: JSON.stringify(credentials),
        }
      );

      const data = await response.json();

      if (data.warehouses) {
        setWarehouse(data.warehouses);
      } else {
        toast({
          title: "Error",
          description: "Failed to fetch warehouses",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch warehouses.",
        variant: "destructive",
      });
    }
  };

  const getTestConnectionButtonVariant = () => {
    switch (connectionStatus) {
      case "success":
        return "outline";
      case "failed":
        return "destructive";
      default:
        return "outline";
    }
  };

  return (
    <div className="shadow-lg border border-[#E0E5EF] rounded-md h-[800px] mb-10 bg-white">
      <div className="flex items-center gap-4 w-[600px] bg-[#EDEFF2] pl-2 mb-6 rounded-t-md">
        <Image className="m-1 p-1" src={Snowflake} alt="Snowflake" width={40} height={50} />
        <p className="text-lg font-semibold text-[#233E7D]">{projectName}</p>
      </div>
      <div className="p-4 rounded-b-md bg-white h-[650px]">
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
            <FormField
              control={form.control}
              name="accountName"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Account Name</FormLabel>
                  <FormControl>
                    <Input
                      type="text"
                      placeholder="Enter Account Name"
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
              name="username"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Username</FormLabel>
                  <FormControl>
                    <Input
                      type="text"
                      placeholder="Enter Username"
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
              name="password"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Password</FormLabel>
                  <FormControl>
                    <Input
                      type="password"
                      placeholder="Enter Password"
                      className="border-[#C8C8C8] rounded-md px-3 py-2 text-base"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage className="text-xs text-[#D82026]" />
                </FormItem>
              )}
            />
            <Button
              type="button"
              variant={getTestConnectionButtonVariant()}
              onClick={(event) => {
                event.preventDefault();
                testConnection();
              }}
              className={cn(
                "rounded-md text-sm font-medium h-10 px-4 py-2",
                connectionStatus === "success" && "border-green-600 text-green-700",
                connectionStatus === "failed" && "border-red-600 text-red-700",
                connectionStatus === "idle" && "border-gray-400 text-gray-700"
              )}
            >
              <Unplug />
              <span className="ml-2">Test Connection</span>
            </Button>
            <FormField
              control={form.control}
              name="warehouse"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-sm font-semibold text-[#233E7D]">Warehouse</FormLabel>
                  <FormControl>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" className="border-[#C8C8C8] rounded-md">
                          {field.value.name || "Select Warehouse"}
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="w-56">
                        <DropdownMenuLabel>Warehouses</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        <DropdownMenuGroup>
                          {warehouse.map((sub) => (
                            <DropdownMenuItem
                              key={sub.name}
                              onClick={() => field.onChange(sub)}
                            >
                              {sub.name}
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
                        initialFocus
                      />
                    </PopoverContent>
                  </Popover>
                  <FormMessage className="text-xs text-[#D82026]" />
                </FormItem>
              )}
            />
            <Button
              variant="default"
              type="submit"
              className="bg-[#D82026] text-white rounded-md hover:bg-[#b81a1f] text-sm font-medium h-10 px-4 py-2"
            >
              Confirm
            </Button>
          </form>
        </Form>
      </div>
    </div>
  );
};

export default SnowflakeForm;
