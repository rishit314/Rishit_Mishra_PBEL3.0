// Allow TypeScript to recognize standard CSS and CSS Module imports
declare module "*.css" {
  const content: { [className: string]: string };
  export default content;
}

declare module "*.scss";
declare module "*.sass";
declare module "*.less";

// Allow image imports just in case you need them later
declare module "*.png";
declare module "*.jpg";
declare module "*.jpeg";
declare module "*.svg";
declare module "*.gif";
declare module "*.webp";