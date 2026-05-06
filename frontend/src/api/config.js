export const IRRIGATION_API_BASE_URL =
  import.meta.env.VITE_IRRIGATION_API_BASE_URL || "/api";

export const DISEASE_API_BASE_URL =
  import.meta.env.VITE_DISEASE_API_BASE_URL || IRRIGATION_API_BASE_URL;
