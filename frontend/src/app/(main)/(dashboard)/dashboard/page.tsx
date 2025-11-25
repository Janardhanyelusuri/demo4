"use client";

import React, { useState } from "react";
import Welcome from "@/assets/welcome.svg";
import LandingPage from "@/assets/landingPage.svg";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

type Props = {};

function CreateProject() {
  const [projectName, setProjectName] = useState("");
  const [inputError, setInputError] = useState(false);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const router = useRouter();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value.replace(/\s/g, "");
    setProjectName(newValue);
  };

  const handleContinue = () => {
    if (projectName.trim() === "") {
      setInputError(true);
      return;
    }

    const url = `/dashboardOnboarding?dashboardName=${encodeURIComponent(
      projectName.trim()
    )}`;
    router.push(url);
  };

  const toggleDialog = () => {
    setIsDialogOpen(!isDialogOpen);
    if (!isDialogOpen) {
      setInputError(false);
    }
  };

  return (
  <Dialog open={isDialogOpen} onOpenChange={toggleDialog}>
    <DialogTrigger asChild>
      <Button
        variant="default"
        onClick={toggleDialog}
        className="bg-[#0066CC] hover:bg-[#0055CC] text-white"
      >
        Create Dashboard
      </Button>
    </DialogTrigger>
    <DialogContent className="sm:max-w-[550px]">
      <DialogHeader>
        <DialogTitle className="text-[#0066CC]">New Dashboard</DialogTitle>
        <DialogDescription>
          Enter the name of your new Dashboard. Click continue when
          you&apos;re done.
        </DialogDescription>
      </DialogHeader>
      <div className="grid gap-4 py-4">
        <div className="grid grid-cols-1 items-center gap-4">
          <Input
            id="project-name"
            value={projectName}
            onChange={handleInputChange}
            className={`w-full ${inputError ? "border-[#FF3B30]" : ""}`}
            placeholder="Name of Your Dashboard"
          />
          <p className="text-xs text-[#8E8E93]">
            Note: Spaces are not allowed. You can use hyphens (-) and
            underscores (_).
          </p>
        </div>
        {inputError && (
          <p className="text-[#FF3B30] text-xs">Dashboard name is required</p>
        )}
      </div>
      <DialogFooter>
        <Button
          variant="outline"
          onClick={toggleDialog}
          className="border-[#0066CC] text-[#0066CC] hover:bg-[#0066CC]/10"
        >
          Cancel
        </Button>
        <Button
          variant="default"
          onClick={handleContinue}
          className="bg-[#0066CC] hover:bg-[#0055CC] text-white"
        >
          Continue
        </Button>
      </DialogFooter>
    </DialogContent>
  </Dialog>
);
}

const Page = (props: Props) => {
  const router = useRouter();

  const handleExistingProjectClick = () => {
    router.push("/dashboard-home");
  };

  return (
    <div className="flex justify-center items-center p-16 bg-[#F1F4FF] h-screen">
      <div>
        <Image src={Welcome} alt="Welcome to Cloud Pulse" />
        <div className="mt-8 flex gap-5">
          <CreateProject />
          <Button
            variant="default"
            onClick={handleExistingProjectClick}
            className="bg-[#D82026] hover:bg-[#b81a1f] text-white"
          >
            Existing Dashboard
          </Button>
        </div>
      </div>
      <div>
        <Image src={LandingPage} alt="Landing Page" />
      </div>
    </div>
  );
};

export default Page;
