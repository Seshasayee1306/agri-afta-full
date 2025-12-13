cat > frontend/Dockerfile << 'EOF'
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Build the production bundle
RUN npm run build

EXPOSE 3000

# Use preview to serve the built app
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "3000"]
EOF