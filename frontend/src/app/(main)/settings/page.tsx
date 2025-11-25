import React from "react";

type Props = {};

const Page = (props: Props) => {
  return (
    <div className="bg-[#F9FEFF] fixed inset-0 flex flex-col items-center justify-center overflow-hidden">
      <h1 className="text-3xl font-bold text-center mb-4 text-[#233E7D]">
        Coming Soon!
      </h1>
      <p className="text-base text-[#6B7280] text-center">
        Work in Progress
      </p>
    </div>
  );
};

export default Page;
