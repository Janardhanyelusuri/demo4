module.exports = {
    extendContext: (req) => {
        console.log(req.securityContext);
        return {
            securityContext: {
                ...req.securityContext,
            }
        }
    },
  contextToAppId: ({ securityContext }) =>
        `CUBE_APP_${securityContext.schemaName}_${securityContext.tagsBudget}`,
//    `CUBE_APP_${securityContext.schemaName}`,

};
