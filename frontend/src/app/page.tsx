"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    router.push("/landingpage");
  }, [router]);

  return <div className="bg-[#F9FEFF] min-h-screen" />;
}
