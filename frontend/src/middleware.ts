import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { getToken } from "next-auth/jwt";

export async function middleware(request: NextRequest) {
  const token = await getToken({
    req: request,
    secret: process.env.AUTH_SECRET,
  });

  const path = request.nextUrl.pathname;

  // Allow access to the login page
  if (path === "/login" && !token) {
    return NextResponse.next();
  }

  // If user is not logged in and trying to access a protected route, redirect to login page
  if (!token && path !== "/login") {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  // If user is logged in and trying to access the login page, redirect to landing page
  if (token && path === "/login") {
    return NextResponse.redirect(new URL("/landingpage", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    // Add paths to protect
    "/",
    "/landingpage/:path*",
    "/onboarding/:path*",
    "/cloudOnboardingForm/:path*",
    "/BreadcrumbNavigation/:path*",
    "/projects/:path*",
    "/notification/:path*",
    "/settings/:path*",
    "/dashboard/:path*", // Protecting dashboard routes
    "/dashboard-home/:path*", // Protecting dashboard-home routes
    "/dashboardOnboarding/:path*", // Protecting dashboardOnboarding routes
  ],
};