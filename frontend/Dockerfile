# Use an official Node.js runtime as a parent image
FROM node:14

# Set the working directory in the container
WORKDIR /frontend

# Copy the package.json and package-lock.json files into the container
COPY package.json package.json

# Install any needed packages specified in package.json
RUN npm install

# Copy the rest of the frontend code into the container
COPY . .

# Make port 3000 available to the world outside this container
EXPOSE 3000

# Run the application
CMD ["npm", "start"]