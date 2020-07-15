$(function() {
    $("input").on("change paste keyup", update);
});

function update() {
    console.log("Update");
    // Read Input Parameters
    let C2 = parseFloat($(".C2").val());
    let E2 = parseFloat($(".E2").val());
    let G2 = parseFloat($(".G2").val());

    // Compute Dependents
    let E3 = Math.floor(C2*E2);
    let C3 = (1-C2)*C2*E2/(C2*E2-E3)/(1+E3-C2*E2);
    let G3 = 1/C3;

    // Read Design Parameters
    let C5 = parseFloat($(".C5").val());
    let E5 = parseFloat($(".E5").val());
    let G5 = parseFloat($(".G5").val());
    let C6 = parseFloat($(".C6").val());
    let E6 = parseFloat($(".E6").val());
    let G6 = parseFloat($(".G6").val());

    // Compute Design Parameters
    let C7 = C6/C5;
    let E7 = -E6/E5;
    let G7 = G6/G5;

    // Compute Inductance Dual Model Parameters
    let C10 = C5;
    let C11 = C6;
    let C12 = Math.pow(G2, 2)/(C5+E2*C6);
    let C13 = Math.pow(G2, 2)*(E2-1)*C6/C5/(C5+E2*C6);
    let C14 = Math.pow(G2, 2)*(C5+(E2-1)*C6)/C5/(C5+E2*C6);
    let C15 = -1*Math.pow(G2, 2)*C6/C5/(C5+E2*C6);
    let C16 = 1/C5;
    let C17 = 1/C6;
    let C18 = (C5+(E2-1)*C6)/C5/(C5+E2*C6);
    let C19 = 1/(C5/E2+C6);

    // Compute Inductance Matrix Model Parameters
    let E10 = Math.pow(G2, 2)/(E5-E6);
    let E11 = -1*Math.pow(G2, 2)*E6/(E5-E6)/(E5+(E2-1)*E6);
    let E12 = E5+(E2-1)*E6;
    let E13 = -(E2-1)*E6;
    let E14 = E5;
    let E15 = E6;
    let E16 = 1/E10;
    let E17 = 1/E11;
    let E18 = (E10+(E2-1)*E11)/E10/(E10+E2*E11);
    let E19 = 1/(E10/E2+E11);

    // Compute Multiwinding Transformer Model Parameters
    let G10 = Math.pow(G2, 2)*(E2-1)/((E2-1)*G5+E2*G6);
    let G11 = Math.pow(G2, 2)*G6/G5/((E2-1)*G5+E2*G6);
    let G12 = G5;
    let G13 = G6;
    let G14 = G5+G6;
    let G15 = -1*G6/(E2-1);
    let G16 = 1/G10;
    let G17 = 1/G11;
    let G18 = (G10+(E2-1)*G11)/G10/(G10+E2*G11);
    let G19 = 1/(G10/E2+G11);

    // Compute Loss, Lpss, Lotr, Lptr, Ripple Ratio
    let C20 = (1-C2)*C2*E2*Math.pow(G2, 2)/(C5+E2*C6)/(E3+1-C2*E2)/(C2*E2-E3);
    let C21 = Math.pow(G2, 2)*(1-C2)/(-1*Math.pow(E3, 2)*C6/C2/E2-E3*C6/C2/E2+2*E3*C6-C2*E2*C6+C6-C2*C5+C5);
    let C22 = Math.pow(G2, 2)/E2/(C5+E2*C6);
    let C23 = Math.pow(G2, 2)/(C5+E2*C6);
    let C24 = (-1*Math.pow(E3, 2)*C7/C2/E2-E3*C7/C2/E2+2*E3*C7-C2*E2*C7+C7-C2+1)/(1-C2)/(1+E2*C7);
    let E20 = (1-C2)*C2*E2*(E5+E6*(E2-1))/(C2*E2-E3)/(1+E3-C2*E2);
    let E21 = (E5-E6)*(E5+(E2-1)*E6)/(E5+((E2-2*E3-2)+E3*(E3+1)/C2/E2+(C2*E2*(E2-2*E3-1)+E3*(E3+1))/E2/(1-C2))*E6);
    let E22 = (E5+(E2-1)*E6)/E2;
    let E23 = E5+(E2-1)*E6;
    let E24 = (1-((E2-2*E3-2)+E3*(E3+1)/C2/E2+(C2*E2*(E2-2*E3-1)+E3*(E3+1))/E2/(1-C2))*E7)/(1+E7);
    let G20 = (1-C2)*C2*E2*G5/(C2*E2-E3)/(1+E3-C2*E2);
    let G21 = C2*E2*G5*(1-C2)*(G5*(E2-1)+G6*E2)/(C2*E2*(1-C2)*(E2-1)*G5+(C2*E2*(1-C2*E2)-Math.pow(E3, 2)-E3+2*E3*C2*E2)*G6);
    let G22 = G5/E2;
    let G23 = G5;
    let G24 = (C2*E2*(1-C2)*(E2-1)+(C2*E2*(1-C2*E2)-Math.pow(E3, 2)-E3+2*C2*E2*E3)*G7)/C2/E2/(1-C2)/(E2-1+E2*G7);

    // Update numbers
    $(".E3").text(E3);
    $(".C3").text(C3);
    $(".G3").text(G3);
    $(".C7").text(C7);
    $(".E7").text(E7);
    $(".G7").text(G7);
    $(".C10").text(C10);
    $(".C11").text(C11);
    $(".C12").text(C12);
    $(".C13").text(C13);
    $(".C14").text(C14);
    $(".C15").text(C15);
    $(".C16").text(C16);
    $(".C17").text(C17);
    $(".C18").text(C18);
    $(".C19").text(C19);
    $(".E10").text(E10);
    $(".E11").text(E11);
    $(".E12").text(E12);
    $(".E13").text(E13);
    $(".E14").text(E14);
    $(".E15").text(E15);
    $(".E16").text(E16);
    $(".E17").text(E17);
    $(".E18").text(E18);
    $(".E19").text(E19);
    $(".G10").text(G10);
    $(".G11").text(G11);
    $(".G12").text(G12);
    $(".G13").text(G13);
    $(".G14").text(G14);
    $(".G15").text(G15);
    $(".G16").text(G16);
    $(".G17").text(G17);
    $(".G18").text(G18);
    $(".G19").text(G19);
    $(".C20").text(C20);
    $(".C21").text(C21);
    $(".C22").text(C22);
    $(".C23").text(C23);
    $(".C24").text(C24);
    $(".E20").text(E20);
    $(".E21").text(E21);
    $(".E22").text(E22);
    $(".E23").text(E23);
    $(".E24").text(E24);
    $(".G20").text(G20);
    $(".G21").text(G21);
    $(".G22").text(G22);
    $(".G23").text(G23);
    $(".G24").text(G24);
}
