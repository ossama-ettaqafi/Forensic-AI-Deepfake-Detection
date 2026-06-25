# Stage 1: Build Vite React App
FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies using cache mounts
COPY package*.json ./
RUN npm ci

# Copy code and build production assets
COPY . .
RUN npm run build

# Stage 2: Serve using Nginx
FROM nginx:alpine-slim

# Copy build output to Nginx default html path
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
