FROM node:18 AS build-stage
ENV NODE_ENV=development
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
COPY .env.development .env.production
RUN npm run build

FROM nginx:alpine AS production-stage
RUN apk add --no-cache openssl && \
    mkdir -p /etc/nginx/ssl && \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
      -subj "/CN=localhost" \
      -keyout /etc/nginx/ssl/localhost.key \
      -out /etc/nginx/ssl/localhost.crt

WORKDIR /usr/share/nginx/html
COPY --from=build-stage /app/dist .

COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 443
CMD ["nginx", "-g", "daemon off;"]
