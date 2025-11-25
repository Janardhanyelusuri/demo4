"use client";

import React, { useState } from "react";
import {
  format,
  subMonths,
  subDays,
  isAfter,
  isBefore,
  startOfDay,
} from "date-fns";
import AWS from "@/assets/AWS.svg";
import { Unplug, CalendarIcon } from "lucide-react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Checkbox } from "@/components/ui/checkbox";
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
import { useRouter } from "next/navigation";
import Image from "next/image";
import { createProject, submitAwsFormData, testAwsConnection } from "@/lib/api";


interface AwsFormProps {
 projectName: string | null;
 providerName: string;
 onBack: () => void; // Add onBack prop
}


const formSchema = z.object({
 accessKey: z.string().min(1, { message: "Access key is required." }),
 secretKey: z.string().min(1, { message: "Secret key is required." }),
 fromDate: z.date({
   required_error: "From date is required.",
 }),
 yearlyBudget: z
   .string()
   .regex(/^\d+$/, { message: "Enter a valid number." })
   .optional(),
 export: z.boolean().default(false),
 exportLocation: z.string().optional(),
});


const AwsForm: React.FC<AwsFormProps> = ({ projectName, providerName, onBack }) => { // Include onBack prop
 const { toast } = useToast();
 const router = useRouter();
 const [connectionStatus, setConnectionStatus] = useState<"idle" | "success" | "failed">("idle");
 const today = startOfDay(new Date());
 const yesterday = subDays(today, 1);
 const twelveMonthsAgo = subMonths(yesterday, 12);


 const form = useForm<z.infer<typeof formSchema>>({
   resolver: zodResolver(formSchema),
   defaultValues: {
     accessKey: "",
     secretKey: "",
     fromDate: yesterday,
     yearlyBudget: "",
     export: false,
     exportLocation: "",
   },
 });


 const onSubmit = async (values: z.infer<typeof formSchema>) => {
   const { accessKey, secretKey } = values;


   try {
     const testData = await testAwsConnection(accessKey, secretKey);


     if (!testData.status) {
       toast({
         title: "Connection Failed",
         description:
           testData.message ||
           "Failed to connect to AWS. Please check your credentials.",
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


     const awsRequestBody = {
       aws_access_key: values.accessKey,
       aws_secret_key: values.secretKey,
       date: format(values.fromDate, "yyyy-MM-dd"),
       project_id: projectId,
       monthly_budget: values.yearlyBudget,
       yearly_budget: values.yearlyBudget,
       quarterly_budget: values.yearlyBudget,
       export: values.export,
       status: false,
       export_location: values.exportLocation,
     };


     await submitAwsFormData(awsRequestBody);


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
   const { accessKey, secretKey } = form.getValues();
   if (!accessKey || !secretKey) {
     toast({
       title: "Error",
       description: "Please enter both Access Key and Secret Key.",
       variant: "destructive",
     });
     return;
   }


   try {
     const data = await testAwsConnection(accessKey, secretKey);


     if (data.status) {
       setConnectionStatus("success");
     } else {
       setConnectionStatus("failed");
       toast({
         title: "Connection Failed",
         description: data.message || "Failed to connect to AWS.",
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
   <div className="shadow-lg border border-[#E0E5EF] rounded-md h-[1000px] mb-9 bg-white">
     <div className="flex items-center gap-4 w-[600px] bg-[#EDEFF2] pl-2 mb-6 rounded-t-md">
       <Image className="mt-1" src={AWS} alt="AWS" width={40} height={50} />
       <p className="text-lg font-semibold text-[#233E7D]">{projectName}</p>
     </div>
     <div className="p-4 rounded-b-md bg-white h-[950px]">
       <Form {...form}>
         <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
           <FormField
             control={form.control}
             name="accessKey"
             render={({ field }) => (
               <FormItem>
                 <FormLabel className="text-sm font-semibold text-[#233E7D]">Access Key</FormLabel>
                 <FormControl>
                   <Input
                     type="text"
                     placeholder="Enter Access Key"
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
             name="secretKey"
             render={({ field }) => (
               <FormItem>
                 <FormLabel className="text-sm font-semibold text-[#233E7D]">Secret Key</FormLabel>
                 <FormControl>
                   <Input
                     type="password"
                     placeholder="Enter Secret Key"
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
             onClick={testConnection}
             className="bg-[#D82026] text-white rounded-md hover:bg-[#b81a1f] text-sm font-medium h-10 px-4 py-2"
           >
             <Unplug />
             <span className="ml-2">Test Connection</span>
           </Button>
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
           {form.watch("export") && (
             <FormField
               control={form.control}
               name="exportLocation"
               render={({ field }) => (
                 <FormItem>
                   <FormLabel className="text-sm font-semibold text-[#233E7D]">S3 Location</FormLabel>
                   <FormControl>
                     <Input
                       type="text"
                       placeholder="Enter S3 Location"
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


export default AwsForm;