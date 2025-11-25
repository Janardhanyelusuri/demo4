# Use Node.js LTS version (you can specify your preferred version)
FROM node:22.6.0

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json (if available)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the Next.js app
RUN npm run build

# Expose the port that Next.js runs on (default is 3000)
EXPOSE 3000

# Command to run the Next.js app in production mode
CMD ["npm", "run", "start"]
