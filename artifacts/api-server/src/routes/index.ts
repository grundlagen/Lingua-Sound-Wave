import { Router, type IRouter } from "express";
import healthRouter from "./health";
import homophonesRouter from "./homophones";
import reservoirRouter from "./reservoir";
import flitRouter from "./flit";
import meaningRouter from "./meaning";

const router: IRouter = Router();

router.use(healthRouter);
router.use(homophonesRouter);
router.use(reservoirRouter);
router.use(flitRouter);
router.use(meaningRouter);

export default router;
