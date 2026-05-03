import { Router, type IRouter } from "express";
import healthRouter from "./health";
import homophonesRouter from "./homophones";

const router: IRouter = Router();

router.use(healthRouter);
router.use(homophonesRouter);

export default router;
