// import NextAuth from "next-auth";
// import AzureADProvider from "next-auth/providers/azure-ad";

// export const authOptions = {
//   providers: [
//     AzureADProvider({
//       clientId: process.env.AUTH_MICROSOFT_ENTRA_ID_ID!,
//       clientSecret: process.env.AUTH_MICROSOFT_ENTRA_ID_SECRET!,
//       tenantId: process.env.AUTH_MICROSOFT_ENTRA_ID_TENANT_ID!,
//     }),
//   ],
//   secret: process.env.AUTH_SECRET,
//   pages: {
//     signIn: "/login",
//   },
//   callbacks: {
//     async redirect({ url, baseUrl }: { url: string; baseUrl: string }) {
//       // Allows relative callback URLs
//       if (url.startsWith("/")) return `${baseUrl}${url}`;
//       // Allows callback URLs on the same origin
//       else if (new URL(url).origin === baseUrl) return url;
//       return baseUrl;
//     },
//   },
// };

// const handler = NextAuth(authOptions);

// export { handler as GET, handler as POST };


import NextAuth from "next-auth";
import AzureADProvider from "next-auth/providers/azure-ad";

const handler = NextAuth({
  providers: [
    AzureADProvider({
      clientId: process.env.AUTH_MICROSOFT_ENTRA_ID_ID!,
      clientSecret: process.env.AUTH_MICROSOFT_ENTRA_ID_SECRET!,
      tenantId: process.env.AUTH_MICROSOFT_ENTRA_ID_TENANT_ID!,
      authorization: {
        params: {
          scope: `openid profile email ${process.env.AUTH_MICROSOFT_ENTRA_ID_ID_SCOPE}`,
        },
      },
    }),
  ],
  secret: process.env.AUTH_SECRET,
  pages: {
    signIn: "/login",
  },
  callbacks: {
    async jwt({ token, account }) {
      if (account) {
        token.accessToken = account.access_token; // Store access token on JWT
      }
      return token;
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken; // Attach access token to session
      return session;
    },
  },  
});

export { handler as GET, handler as POST };